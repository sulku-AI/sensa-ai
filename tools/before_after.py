"""M4 tanıtım: kansız önce / sonra klibi (Instagram). Yalnız vekil ve negatif videolar.

    python -m tools.before_after data/vekil/balon.mp4 previews/balon_once_sonra.mp4 --proxy

Sol: orijinal, sağ: bulanıklaştırılmış. data/pozitif/ altındaki bir video verilirse REDDEDER (D-001):
tanıtım klipleri yalnız kansız içerikle yapılır. Dikey (9:16) çıktı için --vertical.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from sensa_ai import blur, config, pipeline
from sensa_ai.blurtrack import BlurTrack

ROOT = Path(__file__).resolve().parents[1]


def guard(video: Path, data_root: Path = ROOT / "data") -> None:
    if (data_root / "pozitif").resolve() in video.resolve().parents:
        raise PermissionError("tanıtım klibi pozitif (kanlı) videodan yapılmaz (D-001)")


def label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def make(video: Path, out: Path, bt: BlurTrack, vertical: bool = False) -> Path:
    guard(video)
    b = config.load()["blur"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h * 2) if vertical else (w * 2, h)
    out.parent.mkdir(parents=True, exist_ok=True)
    silent = out.with_name(out.stem + ".sessiz.mp4")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        regions = bt.active_regions(i / fps, b["ramp_in_s"], b["ramp_out_s"])
        after = blur.apply(
            frame, blur.region_mask((h, w), regions, int(b["dilate_px"]), int(b["feather_px"])), b["sigma_px"]
        )
        before = frame.copy()
        label(before, "once")
        after = after.copy()
        label(after, "sonra")
        vw.write(np.vstack([before, after]) if vertical else np.hstack([before, after]))
        i += 1
    cap.release()
    vw.release()
    ff = shutil.which("ffmpeg")
    if ff:
        pipeline.mux_audio(ff, silent, video, out)
        silent.unlink(missing_ok=True)
    else:
        silent.replace(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--proxy", action="store_true", help="modelsiz vekil dedektör")
    ap.add_argument("--track", type=Path, default=None, help="hazır .blurtrack.json")
    ap.add_argument("--vertical", action="store_true")
    args = ap.parse_args()
    guard(args.video)
    if args.track:
        bt = BlurTrack.load(args.track)
    elif args.proxy:
        from sensa_ai.detect import ColorProxyDetector

        bt = pipeline.analyze(args.video, ColorProxyDetector(), config.load_prompts()["proxy"])
    else:
        from sensa_ai import models

        bt = pipeline.analyze(args.video, models.build_default(use_gate=False), config.load_prompts()["proxy"])
    print("->", make(args.video, args.out, bt, args.vertical))


if __name__ == "__main__":
    main()
