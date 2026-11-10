"""Kırpım doğrulayıcı (Y3, D-023): dedektörün bulduğu bölgeye ikinci soru, "kan mı, kırmızı ama kan değil mi".

Dedektör yüksek yakalama için düşük eşikle çalışır; her bölge kırpılır, bir görüntü gömmesi
(SigLIP, `models.SiglipEmbedder`) çıkarılır ve küçük bir doğrusal sınıflandırıcı karar verir.
Sınıflandırıcı `tools/train_verifier` ile eldeki pozitif ve zor negatif kırpımlardan eğitilir.
Bu modül yalnız numpy kullanır; gömme işlevi dışarıdan verilir (testlerde sahte gömme).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .blurtrack import Box
from .detect import Detection, Detector

Embed = Callable[[list[np.ndarray]], np.ndarray]  # BGR kırpımlar -> (n, d) gömme


def crop(frame: np.ndarray, box: Box, margin: float) -> np.ndarray:
    """Normalize kutuyu `margin` (kutu boyutunun oranı) kadar genişletip keser: bağlam ayırt etmeye yarar."""
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, y0 = int(max(0.0, box[0] - margin * bw) * w), int(max(0.0, box[1] - margin * bh) * h)
    x1, y1 = int(min(1.0, box[2] + margin * bw) * w), int(min(1.0, box[3] + margin * bh) * h)
    return frame[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)]


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@dataclass
class Head:
    """Birim uzunluğa getirilmiş gömme üzerinde lojistik regresyon."""

    w: np.ndarray
    b: float

    def prob(self, emb: np.ndarray) -> np.ndarray:
        z = _unit(emb) @ self.w + self.b
        return np.asarray(1.0 / (1.0 + np.exp(-z)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w, b=self.b)

    @classmethod
    def load(cls, path: str | Path) -> Head:
        d = np.load(path)
        return cls(d["w"], float(d["b"]))


def fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 2000, lr: float = 2.0) -> Head:
    """Sınıf dengeli lojistik regresyon, düz gradyan inişi (ek bağımlılık yok). y: 1 = kan."""
    x, y = _unit(x), np.asarray(y, np.float64)
    wt = np.where(y > 0, 0.5 / max(1.0, y.sum()), 0.5 / max(1.0, (1 - y).sum()))  # iki sınıf eşit ağırlık
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(iters):
        g = (1.0 / (1.0 + np.exp(-(x @ w + b))) - y) * wt
        w -= lr * (x.T @ g + l2 * w)
        b -= lr * float(g.sum())
    return Head(w, b)


class VerifiedDetector:
    """Dedektörü sarar: doğrulayıcı olasılığı `threshold` altındaki bölgeler atılır. `Detector` arayüzü."""

    def __init__(self, detector: Detector, embed: Embed, head: Head, threshold: float, margin: float) -> None:
        self.detector, self.embed, self.head = detector, embed, head
        self.threshold, self.margin = threshold, margin

    def __call__(self, frame_bgr: np.ndarray, prompts: list[str]) -> list[Detection]:
        dets = self.detector(frame_bgr, prompts)
        if not dets:
            return dets
        probs = self.head.prob(self.embed([crop(frame_bgr, d.box, self.margin) for d in dets]))
        return [d for d, p in zip(dets, probs, strict=True) if p >= self.threshold]

    def __getattr__(self, name: str) -> Any:  # box_threshold, name vb. sarılan dedektörden
        return getattr(self.__dict__["detector"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("detector", "embed", "head", "threshold", "margin"):
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["detector"], name, value)
