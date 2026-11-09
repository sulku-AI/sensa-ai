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
    poly: list[list[float]] | None = None  # takip maskesinden, normalize


class Detector(Protocol):
    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]: ...


class ColorProxyDetector:
    """Model gerektirmeyen vekil dedektör: belirgin doygun kırmızı alanları kutular.

    Gerçek kan tespiti DEĞİLDİR. Yalnız boru hattını (iz, bulanıklık, ileri bakma)
    kansız vekil videolarla (kırmızı balon, D-005) ve CI'da sınamak içindir.

    `scored=True` iken skor doygunluktan gelir (S 140 -> 0, 255 -> 1), `min_score` altı atılır: ses ipucu
    demosunda (D-019) "zayıf / orta / emin" görüntü bölgelerini kansız taklit etmek için.
    """

    def __init__(self, min_area_frac: float = 0.002, scored: bool = False, min_score: float = 0.0) -> None:
        self.min_area_frac, self.scored, self.min_score = min_area_frac, scored, min_score

    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        import cv2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lo = cv2.inRange(hsv, (0, 150, 70), (10, 255, 255))
        hi = cv2.inRange(hsv, (170, 150, 70), (180, 255, 255))
        mask = cv2.bitwise_or(lo, hi)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        h, w = mask.shape
        out: list[Detection] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area >= self.min_area_frac * h * w:
                comp = labels == i
                score = min(1.0, max(0.0, (float(hsv[..., 1][comp].mean()) - 140) / 115)) if self.scored else 1.0
                if score < self.min_score:
                    continue
                box = (x / w, y / h, (x + bw) / w, (y + bh) / h)
                out.append(Detection("proxy", box, score, _component_poly(comp, w, h)))
        return out


def _component_poly(comp: np.ndarray, w: int, h: int) -> list[list[float]] | None:
    """Bileşenin dış sınırı, sadeleştirilmiş normalize poligon (bulanıklık kutu değil şekli izlesin)."""
    import cv2

    cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cs:
        a = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(a) >= 3:
            polys.append([float(v) for x, y in a for v in (x / w, y / h)])
    return polys or None
