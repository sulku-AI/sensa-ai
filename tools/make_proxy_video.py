"""Kansız vekil test videosu (D-005): kırmızı bir "balon" yeşil arka planda gezinir.

    python -m tools.make_proxy_video data/vekil/balon.mp4 --seconds 20

Balon 4. saniyede görünür, 14. saniyede kaybolur; arada bir kez 1 sn saklanır (takip ve
yeniden bulma sınansın). Emirhan bu videoyu rahatça izleyebilir.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def make(out: Path, seconds: float = 20.0, fps: int = 25, w: int = 1280, h: int = 720) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = (70, 120, 60)
    bg = cv2.add(bg, rng.integers(0, 30, (h, w, 3), dtype=np.uint8))
    for i in range(int(seconds * fps)):
        t = i / fps
        f = bg.copy()
        cv2.rectangle(f, (int(w * 0.6), int(h * 0.55)), (int(w * 0.8), int(h * 0.95)), (40, 60, 90), -1)  # kutu
        if 4.0 <= t < 14.0 and not (9.0 <= t < 10.0):
            x = int(w * (0.2 + 0.5 * (t - 4) / 10))
            y = int(h * (0.4 + 0.1 * math.sin(t * 2)))
            cv2.circle(f, (x, y), 60, (30, 30, 210), -1)
            cv2.line(f, (x, y + 60), (x, y + 200), (220, 220, 220), 2)
        cv2.putText(f, f"{t:5.1f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        vw.write(f)
    vw.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    print("->", make(args.out, args.seconds))


if __name__ == "__main__":
    main()
