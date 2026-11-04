"""Tespit arayüzü. Gerçek modeller (SigLIP, Grounding DINO, OWLv2, SAM 2) M1'de
bu arayüzün arkasına eklenir; ağır bağımlılıklar `pip install -e ".[models]"` ile gelir.
Model seçimi ve gerekçesi: docs/MODEL_SECIMI.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .blurtrack import Box


@dataclass
class Detection:
    label: str
    box: Box  # normalize
    score: float


class Detector(Protocol):
    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]: ...


class ColorProxyDetector:
    """Model gerektirmeyen vekil dedektör: belirgin doygun kırmızı alanları kutular.

    Gerçek kan tespiti DEĞİLDİR. Yalnız boru hattını (iz, bulanıklık, ileri bakma)
    kansız vekil videolarla (kırmızı balon, D-005) ve CI'da sınamak içindir.
    """

    def __init__(self, min_area_frac: float = 0.002) -> None:
        self.min_area_frac = min_area_frac

    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        import cv2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lo = cv2.inRange(hsv, (0, 150, 70), (10, 255, 255))
        hi = cv2.inRange(hsv, (170, 150, 70), (180, 255, 255))
        mask = cv2.bitwise_or(lo, hi)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        h, w = mask.shape
        out: list[Detection] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area >= self.min_area_frac * h * w:
                out.append(Detection("proxy", (x / w, y / h, (x + bw) / w, (y + bh) / h), 1.0))
        return out
