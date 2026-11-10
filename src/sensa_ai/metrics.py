"""Değerlendirme ölçütleri (H-01, H-02). Yalnız sayı üretir, görüntü üretmez.

Kullanıcı açısından asıl soru "kanlı piksellerin ne kadarı bulanıklaştı": bu yüzden kutu
IoU'su yerine **kapsama** (coverage) = |GT ∩ bulanık| / |GT| ölçülür. Fazla bulanıklık da
ayrıca sayılır, çünkü gereksiz bulanıklık izlemeyi bozar (H-04).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .blurtrack import BlurTrack, Box


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """İkili oran için %95 güven aralığı (Wilson). Küçük setlerde 0.957 = 44/46 gibi sonuçların payı."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, float(c - h)), min(1.0, float(c + h)))


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
        hits = int((cov >= hit_coverage).sum())
        ylo, yhi = wilson(hits, int(cov.size))
        alo, ahi = wilson(self.neg_flagged, self.n_neg)
        return {
            "pozitif_goruntu": self.n_pos,
            "yakalama_orani": float((cov >= hit_coverage).mean()) if cov.size else float("nan"),
            "ortalama_kapsama": float(cov.mean()) if cov.size else float("nan"),
            "yakalama_alt_95": ylo,
            "yakalama_ust_95": yhi,
            "kapsama_p10": float(np.percentile(cov, 10)) if cov.size else float("nan"),
            "fazla_bulaniklik_ort": float(np.mean(self.overblur)) if self.overblur else float("nan"),
            "negatif_goruntu": self.n_neg,
            "negatif_yanlis_alarm_orani": self.neg_flagged / self.n_neg if self.n_neg else float("nan"),
            "yanlis_alarm_alt_95": alo,
            "yanlis_alarm_ust_95": ahi,
            "negatif_bulanik_alan_ort": float(np.mean(self.neg_area)) if self.neg_area else float("nan"),
        }


@dataclass
class Event:
    """Ortağın etiketi: [start, end) aralığında rahatsız edici bölge var. box isteğe bağlı (normalize)."""

    start: float
    end: float
    box: Box | None = None


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def event_scores(track: BlurTrack, events: list[Event], step_s: float = 0.1, **timing: Any) -> dict[str, float]:
    """H-01 video düzeyi: olay yakalama, kare kapsama, geç başlama, olay dışı bulanıklık (H-02).

    Bir an "kapsanmış" sayılır: o anda en az bir bulanık bölge var ve (etiket kutusu varsa) onunla örtüşüyor.
    """
    duration = track.duration_s
    covered_per_event: list[float] = []
    lateness: list[float] = []
    n_steps = int(duration / step_s) + 1
    blurred_outside = 0
    for i in range(n_steps):
        t = i * step_s
        regions = track.active(t, **timing)
        if regions and not any(e.start <= t < e.end for e in events):
            blurred_outside += 1
    for e in events:
        ts = np.arange(e.start, e.end, step_s)
        hits = []
        for t in ts:
            regions = track.active(float(t), **timing)
            hits.append(any(e.box is None or _overlaps(b, e.box) for b, _ in regions))
        covered_per_event.append(float(np.mean(hits)) if len(hits) else 1.0)
        first = next((float(t) for t, h in zip(ts, hits, strict=True) if h), None)
        lateness.append(first - e.start if first is not None else e.end - e.start)
    outside_s = blurred_outside * step_s
    clean_min = max(duration - sum(e.end - e.start for e in events), 1e-9) / 60
    cov = np.asarray(covered_per_event)
    return {
        "olay": len(events),
        "olay_yakalama": float((cov >= 0.5).mean()) if cov.size else float("nan"),
        "kare_kapsama_ort": float(cov.mean()) if cov.size else float("nan"),
        "kare_kapsama_min": float(cov.min()) if cov.size else float("nan"),
        "gec_baslama_ort_s": float(np.mean(lateness)) if lateness else float("nan"),
        "olay_disi_bulanik_s": outside_s,
        "olay_disi_bulanik_s_dk": outside_s / clean_min,
    }
