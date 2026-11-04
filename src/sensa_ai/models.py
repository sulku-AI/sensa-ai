"""Gerçek modeller (M1): kapı, metinle bölge bulma, maske takibi.

Ağır bağımlılıklar yalnız burada ve tembel (lazy) içe aktarılır:
    pip install -e ".[models]"
Model kimlikleri: config/models.yaml. Seçim gerekçesi: docs/MODEL_ARASTIRMA.md (D-011).

Hepsi `from_pretrained` alır; testler (tests/test_models.py) rastgele ağırlıklı küçük
yapılandırmalarla indirme olmadan çalışır.
"""

from __future__ import annotations

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
        max_n = int(config.load()["detect"]["max_regions"])
        order = res["scores"].argsort(descending=True)[:max_n]
        return [
            Detection(
                str(names[i]) if names is not None else "region",
                _clip_box(res["boxes"][i], w, h),
                float(res["scores"][i]),
            )
            for i in order.tolist()
        ]


class SegDetector:
    """Eğitilmiş RF-DETR-Seg kontrol noktası (tools/train_rfdetr). İstemleri yok sayar: tek sınıf "blood".

    `TextDetector` ile aynı arayüz; maskeden poligon da verir (bulanıklık şekli izler).
    """

    def __init__(self, model: Any, device: str, box_threshold: float, name: str) -> None:
        self.model, self.device, self.box_threshold, self.name = model, device, box_threshold, name
        self.text_threshold = box_threshold

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None) -> SegDetector:
        from rfdetr import RFDETRSegNano

        d = dev.pick(device)
        thr = float(config.load()["detect"]["box_threshold"])
        return cls(RFDETRSegNano(pretrain_weights=path, device=d), d, thr, Path(path).name)

    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        from .detect import _component_poly

        h, w = frame_bgr.shape[:2]
        res = self.model.predict(_to_pil(frame_bgr), threshold=self.box_threshold)
        max_n = int(config.load()["detect"]["max_regions"])
        order = np.argsort(-np.asarray(res.confidence))[:max_n]
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


def build_default(device: str | None = None, use_gate: bool = True) -> GatedDetector:
    prompts = config.load_prompts()
    gate = SiglipGate.from_pretrained(device=device) if use_gate else None
    return GatedDetector(
        gate, TextDetector.from_pretrained(device=device), prompts["negative"], config.load()["gate"]["threshold"]
    )
