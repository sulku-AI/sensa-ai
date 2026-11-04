"""Aşama C: kutu isteminden maske ve videoda yayılım (SAM 2.1 / EdgeTAM, Apache 2.0).

transformers'taki Sam2Video / EdgeTamVideo modelleri aynı işlemciyi (Sam2VideoProcessor) kullanır.
M5'te varsayılan EdgeTAM (config/models.yaml, D-012).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import device as dev
from .blurtrack import Box


class Tracker:
    def __init__(self, model: Any, processor: Any, device: str, model_id: str = "") -> None:
        self.model, self.processor, self.device, self.model_id = model, processor, device, model_id

    @classmethod
    def from_pretrained(cls, model_id: str | None = None, device: str | None = None) -> Tracker:
        from .models import resolve

        d = dev.pick(device)
        mid = model_id or resolve("track")
        try:
            return cls(*_load(mid, d), d, mid)
        except (OSError, ValueError) as e:
            fallback = resolve("track", "sam2_tiny")
            if model_id is not None or mid == fallback:
                raise
            print(f"[tracking] {mid} yüklenemedi ({type(e).__name__}: {str(e)[:300]}); yedek {fallback} kullanılıyor")
            return cls(*_load(fallback, d), d, fallback)

    def track(self, frames_bgr: list[np.ndarray], boxes: list[Box], start_idx: int = 0) -> list[np.ndarray]:
        """`start_idx` karesindeki normalize kutulardan başlayıp ileri yayar.

        Dönüş: her kare için bool maske (bütün nesnelerin birleşimi); start_idx öncesi boş.
        """
        per = self.track_objects(frames_bgr, boxes, start_idx)
        if not frames_bgr:
            return []
        h, w = frames_bgr[0].shape[:2]
        return [np.any(np.stack(ms), 0) if ms else np.zeros((h, w), bool) for ms in per]

    def track_objects(
        self, frames_bgr: list[np.ndarray], boxes: list[Box], start_idx: int = 0
    ) -> list[list[np.ndarray]]:
        """Nesne başına maskeler: sonuç[kare][nesne]. start_idx öncesi kareler boş liste."""
        import torch

        if not frames_bgr:
            return []
        h, w = frames_bgr[0].shape[:2]
        if not boxes:
            return [[] for _ in frames_bgr]
        rgb = [np.ascontiguousarray(f[..., ::-1]) for f in frames_bgr]
        session = self.processor.init_video_session(video=rgb, inference_device=self.device, dtype=torch.float32)
        px = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes]
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=start_idx, obj_ids=list(range(1, len(px) + 1)), input_boxes=[px]
        )
        out_masks: list[list[np.ndarray]] = [[] for _ in frames_bgr]
        with torch.no_grad():
            self.model(inference_session=session, frame_idx=start_idx)
            for out in self.model.propagate_in_video_iterator(session, start_frame_idx=start_idx):
                m = self.processor.post_process_masks([out.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
                m = m.reshape(-1, h, w).cpu().numpy().astype(bool)
                out_masks[out.frame_idx] = [m[k] for k in range(m.shape[0])]
        return out_masks

    def bench(self, frames_bgr: list[np.ndarray], box: Box) -> float:
        """Kare başı saniye (ilk kare istemi dahil)."""
        self.track(frames_bgr[:2], [box])  # ısınma
        t0 = time.perf_counter()
        self.track(frames_bgr, [box])
        return (time.perf_counter() - t0) / len(frames_bgr)


def mask_to_polys(mask: np.ndarray, epsilon_frac: float = 0.01, max_polys: int = 4) -> list[list[float]]:
    """Maskeyi sadeleştirilmiş normalize poligonlara çevirir (en büyük `max_polys` parça)."""
    import cv2

    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_polys]
    out = []
    for c in contours:
        approx = cv2.approxPolyDP(c, epsilon_frac * max(h, w), True).reshape(-1, 2)
        if len(approx) >= 3:
            out.append([round(float(v), 4) for xy in approx for v in (xy[0] / w, xy[1] / h)])
    return out


def mask_to_box(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def _load(mid: str, d: str) -> tuple[Any, Any]:
    """Video modeli sınıfını açıkça seç: AutoModel EdgeTAM kimliğinde görüntü modelini seçebilir."""
    from transformers import EdgeTamVideoModel, Sam2VideoModel, Sam2VideoProcessor

    klass = EdgeTamVideoModel if "edgetam" in mid.lower() else Sam2VideoModel
    model = klass.from_pretrained(mid).to(d).eval()
    return model, Sam2VideoProcessor.from_pretrained(mid)
