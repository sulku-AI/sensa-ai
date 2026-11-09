"""Gerçek modeller (M1): kapı, metinle bölge bulma, maske takibi.

Ağır bağımlılıklar yalnız burada ve tembel (lazy) içe aktarılır:
    pip install -e ".[models]"
Model kimlikleri: config/models.yaml. Seçim gerekçesi: docs/MODEL_ARASTIRMA.md (D-011).

Hepsi `from_pretrained` alır; testler (tests/test_models.py) rastgele ağırlıklı küçük
yapılandırmalarla indirme olmadan çalışır.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import config
from . import device as dev
from .blurtrack import Box
from .detect import Detection

MODELS_PATH = config.ROOT / "config" / "models.yaml"


def model_ids(path: Path = MODELS_PATH) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, dict[str, str]] = yaml.safe_load(f)
    return data


def resolve(kind: str, name: str | None = None) -> str:
    ids = model_ids()
    key = name or ids["default"][kind]
    return ids[kind].get(key, key)  # bilinmeyen ad doğrudan HF kimliği sayılır


def _to_pil(frame_bgr: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(frame_bgr[..., ::-1]))


def _clip_box(b: Any, w: float, h: float) -> Box:
    x0, y0, x1, y1 = (float(v) for v in b)
    return (
        min(max(x0 / w, 0.0), 1.0),
        min(max(y0 / h, 0.0), 1.0),
        min(max(x1 / w, 0.0), 1.0),
        min(max(y1 / h, 0.0), 1.0),
    )


class SiglipGate:
    """Aşama A: karede pozitif istemlerden birinin olasılığı (pozitif + negatif üzerinde softmax).

    SigLIP sigmoid ile eğitildi; burada istemler arasında karşılaştırma için softmax kullanıyoruz,
    çünkü soru "en çok hangisine benziyor" (kan mı, ketçap mı). Eşik: params gate.threshold.
    """

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model, self.processor, self.device = model, processor, device

    @classmethod
    def from_pretrained(cls, model_id: str | None = None, device: str | None = None) -> SiglipGate:
        from transformers import AutoModel, AutoProcessor

        d = dev.pick(device)
        mid = model_id or resolve("gate")
        model = AutoModel.from_pretrained(mid, dtype=dev.dtype_for(d)).to(d).eval()
        return cls(model, AutoProcessor.from_pretrained(mid), d)

    def score(self, frame_bgr: np.ndarray, positive: list[str], negative: list[str]) -> float:
        import torch

        texts = [f"a photo of {p}" for p in positive + negative]
        inputs = self.processor(text=texts, images=_to_pil(frame_bgr), padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image[0].float()
        probs = logits.softmax(-1)
        return float(probs[: len(positive)].sum())


class TextDetector:
    """Aşama B: metin istemli kutular. LLMDet / MM-Grounding-DINO / Grounding DINO / OWLv2.

    `family` model yapılandırmasından çıkar: "owlv2" ya da "grounding" (GD, MM-GD, LLMDet).
    """

    def __init__(self, model: Any, processor: Any, device: str, box_threshold: float, text_threshold: float) -> None:
        self.model, self.processor, self.device = model, processor, device
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.max_regions = int(config.load()["detect"]["max_regions"])  # karede bir kez değil, burada okunur
        mt = getattr(model.config, "model_type", "")
        self.family = "owlv2" if mt.startswith("owl") else "grounding"

    @classmethod
    def from_pretrained(cls, model_id: str | None = None, device: str | None = None) -> TextDetector:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        d = dev.pick(device)
        mid = model_id or resolve("detect")
        # Grounding ailesi FP16'da sayısal sorun çıkarabiliyor; önce FP32, bench sonra karar verir.
        model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(d).eval()
        p = config.load()["detect"]
        return cls(model, AutoProcessor.from_pretrained(mid), d, p["box_threshold"], p["text_threshold"])

    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        import torch

        h, w = frame_bgr.shape[:2]
        image = _to_pil(frame_bgr)
        labels = [prompts]
        extra = {"padding": True} if self.family == "owlv2" else {}  # OWL istem uzunlukları farklı olabilir
        inputs = self.processor(images=image, text=labels, return_tensors="pt", **extra).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.family == "owlv2":
            # OWLv2 görüntüyü kareye doldurur: kutular (s, s) üzerinde, s = max(h, w)
            s = max(h, w)
            res = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=[(s, s)], threshold=self.box_threshold, text_labels=labels
            )[0]
        else:
            res = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]
        names = res["text_labels"] if "text_labels" in res else res.get("labels")
        order = res["scores"].argsort(descending=True)[: self.max_regions]
        return [
            Detection(
                str(names[i]) if names is not None else "region",
                _clip_box(res["boxes"][i], w, h),
                float(res["scores"][i]),
            )
            for i in order.tolist()
        ]


@contextlib.contextmanager
def quiet_library_warnings() -> Iterator[None]:
    """Model yüklenirken torch.jit / izleme (TracerWarning) uyarılarını gizler: zararsız ama ekranı dolduruyor
    ve "bir şey mi bozuldu" diye düşündürüyor. Bizim kendi uyarılarımız ve hatalar görünmeye devam eder."""
    import logging

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\..*")
        warnings.filterwarnings("ignore", message=r".*TracerWarning.*")
        warnings.filterwarnings("ignore", message=r"Converting a tensor to a Python .*")
        warnings.filterwarnings("ignore", message=r".*torch\.jit\..* is deprecated.*")
        try:
            from torch.jit import TracerWarning

            warnings.filterwarnings("ignore", category=TracerWarning)
        except ImportError:
            pass
        noisy = [logging.getLogger(n) for n in ("rf-detr", "rfdetr", "torch.distributed.elastic")]
        old = [lg.level for lg in noisy]
        for lg in noisy:
            lg.setLevel(logging.ERROR)
        try:
            yield
        finally:
            for lg, lv in zip(noisy, old, strict=True):
                lg.setLevel(lv)


def _optimize(model: Any) -> None:
    """rfdetr hızlı çıkarım: yeni sürüm `inference`, eski `optimize_for_inference`. Başarısızsa FP32 kalır."""
    import torch

    for name in ("inference", "optimize_for_inference"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            fn(dtype=torch.float16)
            return
        except Exception as e:  # noqa: BLE001 (sürüm/cihaz farkı: hızlandırma isteğe bağlı)
            print(f"[models] FP16 hızlandırma olmadı ({type(e).__name__}: {e}); FP32 sürüyor")
            return


def detector_name(det: TextDetector | SegDetector) -> str:
    return det.name if isinstance(det, SegDetector) else str(det.model.name_or_path)


def make_detector(name: str | None = None, device: str | None = None) -> TextDetector | SegDetector:
    """models.yaml adı, HF kimliği, `rfdetr:<yol>` ya da `.pth` yolu -> dedektör."""
    mid = resolve("detect", name.removeprefix("rfdetr:") if name else None)
    if (name or "").startswith("rfdetr:") or mid.endswith(".pth"):
        try:
            return SegDetector.from_checkpoint(mid, device=device)
        except FileNotFoundError as e:
            if name is not None:  # açıkça istenen model yoksa hata
                raise
            print(f"[models] {e}; OWLv2 kullanılıyor (yavaş: --mask-track ve kapı önerilir)")
            mid = resolve("detect", "owlv2_base")
    return TextDetector.from_pretrained(mid, device=device)


class SegDetector:
    """Eğitilmiş RF-DETR-Seg kontrol noktası (tools/train_rfdetr). İstemleri yok sayar: tek sınıf "blood".

    `TextDetector` ile aynı arayüz; maskeden poligon da verir (bulanıklık şekli izler).
    """

    def __init__(self, model: Any, device: str, box_threshold: float, name: str) -> None:
        self.model, self.device, self.box_threshold, self.name = model, device, box_threshold, name
        self.text_threshold = box_threshold
        self.max_regions = int(config.load()["detect"]["max_regions"])

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None, fp16: bool | None = None) -> SegDetector:
        p = config.load()["detect"]
        ck = Path(path) if Path(path).is_absolute() else config.ROOT / path
        if not ck.exists() and Path(path).exists():
            ck = Path(path)
        if not ck.exists():
            raise FileNotFoundError(f"kontrol noktası yok: {path} (Drive'dan indirip buraya koy)")
        d = dev.pick(device)
        with quiet_library_warnings():
            from rfdetr import RFDETRSegNano

            try:
                model = RFDETRSegNano(pretrain_weights=str(ck), num_classes=1, device=d)
            except (TypeError, ValueError):  # yeni rfdetr: cihaz alanı yok, kendisi seçer
                model = RFDETRSegNano(pretrain_weights=str(ck), num_classes=1)
            use_fp16 = bool(p.get("seg_fp16", False)) if fp16 is None else fp16
            if use_fp16 and d != "cpu":
                _optimize(model)
        return cls(model, d, float(p.get("seg_threshold", p["box_threshold"])), ck.name)

    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        from .detect import _component_poly

        h, w = frame_bgr.shape[:2]
        res = self.model.predict(_to_pil(frame_bgr), threshold=self.box_threshold)
        order = np.argsort(-np.asarray(res.confidence))[: self.max_regions]
        masks = getattr(res, "mask", None)
        return [
            Detection(
                "blood",
                _clip_box(res.xyxy[i], w, h),
                float(res.confidence[i]),
                _component_poly(masks[i], w, h) if masks is not None else None,
            )
            for i in order.tolist()
        ]


class GatedDetector:
    """A + B: kapı geçen karede dedektörü çalıştırır. `Detector` arayüzünü uygular."""

    def __init__(
        self, gate: SiglipGate | None, detector: TextDetector | SegDetector, negative: list[str], threshold: float
    ) -> None:
        self.gate, self.detector, self.negative, self.threshold = gate, detector, negative, threshold
        self.last_gate_score: float | None = None

    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        if self.gate is not None:
            self.last_gate_score = self.gate.score(frame_bgr, prompts, self.negative)
            if self.last_gate_score < self.threshold:
                return []
        return self.detector(frame_bgr, prompts)


def build_default(
    device: str | None = None, use_gate: bool | None = None, detector: str | None = None
) -> GatedDetector:
    """Varsayılan hat. `use_gate` None ise params `gate.enabled`. Eğitilmiş model dosyası yoksa OWLv2'ye düşer."""
    prompts = config.load_prompts()
    det = make_detector(detector, device=device)
    gate_on = bool(config.load()["gate"].get("enabled", True)) if use_gate is None else use_gate
    gate = SiglipGate.from_pretrained(device=device) if gate_on else None
    return GatedDetector(gate, det, prompts["negative"], config.load()["gate"]["threshold"])
