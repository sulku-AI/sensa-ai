"""Değerlendirme ölçütleri (H-01, H-02). Yalnız sayı üretir, görüntü üretmez.

Kullanıcı açısından asıl soru "kanlı piksellerin ne kadarı bulanıklaştı": bu yüzden kutu
IoU'su yerine **kapsama** (coverage) = |GT ∩ bulanık| / |GT| ölçülür. Fazla bulanıklık da
ayrıca sayılır, çünkü gereksiz bulanıklık izlemeyi bozar (H-04).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .blurtrack import Box


def polygons_to_mask(shape: tuple[int, int], polygons: list[list[float]]) -> np.ndarray:
    """COCO poligonları (x1,y1,x2,y2,...) -> bool maske."""
    m = np.zeros(shape, np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def boxes_to_mask(shape: tuple[int, int], boxes: list[Box], dilate_px: int = 0) -> np.ndarray:
    h, w = shape
    m = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        a = (max(0, int(x0 * w) - dilate_px), max(0, int(y0 * h) - dilate_px))
        b = (min(w, int(np.ceil(x1 * w)) + dilate_px), min(h, int(np.ceil(y1 * h)) + dilate_px))
        m[a[1] : b[1], a[0] : b[0]] = True
    return m


@dataclass
class Score:
    """Bir görüntü setinin özeti."""

    n_pos: int = 0
    n_neg: int = 0
    coverages: list[float] = field(default_factory=list)  # pozitif görüntü başına
    overblur: list[float] = field(default_factory=list)  # pozitif: GT dışı bulanık alan / görüntü alanı
    neg_flagged: int = 0  # negatifte en az bir bölge bulunan görüntü
    neg_area: list[float] = field(default_factory=list)  # negatifte bulanık alan oranı

    def add_positive(self, gt: np.ndarray, pred: np.ndarray) -> None:
        self.n_pos += 1
        g = int(gt.sum())
        self.coverages.append(float((gt & pred).sum() / g) if g else 1.0)
        self.overblur.append(float((pred & ~gt).sum() / gt.size))

    def add_negative(self, pred: np.ndarray) -> None:
        self.n_neg += 1
        self.neg_flagged += int(pred.any())
        self.neg_area.append(float(pred.mean()))

    def summary(self, hit_coverage: float = 0.5) -> dict[str, float]:
        cov = np.asarray(self.coverages) if self.coverages else np.zeros(0)
        return {
            "pozitif_goruntu": self.n_pos,
            "yakalama_orani": float((cov >= hit_coverage).mean()) if cov.size else float("nan"),
            "ortalama_kapsama": float(cov.mean()) if cov.size else float("nan"),
            "kapsama_p10": float(np.percentile(cov, 10)) if cov.size else float("nan"),
            "fazla_bulaniklik_ort": float(np.mean(self.overblur)) if self.overblur else float("nan"),
            "negatif_goruntu": self.n_neg,
            "negatif_yanlis_alarm_orani": self.neg_flagged / self.n_neg if self.n_neg else float("nan"),
            "negatif_bulanik_alan_ort": float(np.mean(self.neg_area)) if self.neg_area else float("nan"),
        }
