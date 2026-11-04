"""Testler yalnız sentetik, kansız görüntü üretir (renkli dikdörtgenler)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def red_square_frame(h: int = 120, w: int = 160, x: int = 60, y: int = 40, s: int = 30) -> np.ndarray:
    frame = np.full((h, w, 3), (90, 140, 60), np.uint8)  # yeşilimsi arka plan
    noise = np.random.default_rng(0).integers(0, 40, (h, w, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    if s > 0:
        cv2.rectangle(frame, (x, y), (x + s, y + s), (0, 0, 230), -1)  # BGR kırmızı
    return frame


@pytest.fixture
def proxy_video(tmp_path: Path) -> Path:
    """4 sn, 10 fps; kırmızı kare 1.0-3.0 sn arasında soldan sağa kayar."""
    path = tmp_path / "vekil.mp4"
    w, h, fps = 160, 120, 10
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    for i in range(40):
        t = i / fps
        if 1.0 <= t < 3.0:
            vw.write(red_square_frame(x=20 + int((t - 1.0) * 40)))
        else:
            vw.write(red_square_frame(s=0))
    vw.release()
    return path
