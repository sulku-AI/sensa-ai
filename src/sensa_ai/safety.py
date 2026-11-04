"""Güvenli önizleme (D-004).

Kesin kural: Emirhan'a hiçbir zaman kanlı görüntü gösterilmez. Bu yüzden insanın
bakabileceği HER görüntü (hata ayıklama küçük resmi, rapor görseli, README örneği)
yalnız `safe_preview` ile üretilir: küçültülür, gri tonlanır ve pikselleştirilir.
Ham kareleri diske yazan başka bir fonksiyon bu pakette yoktur.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config


def safe_preview(
    frame: np.ndarray,
    block_px: int | None = None,
    max_side_px: int | None = None,
    grayscale: bool | None = None,
) -> np.ndarray:
    """BGR kareden tanınmayacak kadar kaba bir küçük resim üretir."""
    p = config.load()["safe_preview"]
    block = int(block_px if block_px is not None else p["block_px"])
    side = int(max_side_px if max_side_px is not None else p["max_side_px"])
    gray = bool(p["grayscale"] if grayscale is None else grayscale)

    h, w = frame.shape[:2]
    scale = side / max(h, w)
    small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    if gray:
        small = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    sh, sw = small.shape[:2]
    tiny = cv2.resize(small, (max(1, sw // block), max(1, sh // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(tiny, (sw, sh), interpolation=cv2.INTER_NEAREST)


def write_preview(path: str | Path, frame: np.ndarray) -> Path:
    """Diske insan için görüntü yazmanın tek yolu."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), safe_preview(frame)):
        raise OSError(f"yazılamadı: {out}")
    return out
