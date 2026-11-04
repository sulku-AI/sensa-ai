"""Çevrimdışı boru hattı (M1): video -> .blurtrack.json -> bulanık video.

    python -m sensa_ai.pipeline analyze video.mp4 --proxy
    python -m sensa_ai.pipeline render video.mp4 video.blurtrack.json cikti.mp4

Not: render şimdilik sesi kopyalamaz (OpenCV). Ses ffmpeg ile M1 içinde eklenecek.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from . import blur, config
from .blurtrack import BlurTrack, Box, Keyframe, Track
from .detect import ColorProxyDetector, Detection, Detector
from .tracking import Tracker, mask_to_box


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def associate(tracks: list[Track], t: float, dets: list[Detection], max_gap_s: float, min_iou: float = 0.1) -> None:
    """Açgözlü eşleme: her tespit en çok örtüşen açık ize eklenir, yoksa yeni iz açar.

    SAM 2 gelince (M1) bu basit kutu takibinin yerini maske yayılımı alır.
    """
    open_tracks = [tr for tr in tracks if t - tr.end <= max_gap_s]
    for d in dets:
        best = max(open_tracks, key=lambda tr: iou(tr.keyframes[-1].box, d.box), default=None)
        if best is not None and iou(best.keyframes[-1].box, d.box) >= min_iou and best.end < t:
            best.keyframes.append(Keyframe(t, d.box, d.score))
        else:
            tr = Track(d.label, [Keyframe(t, d.box, d.score)])
            tracks.append(tr)
            open_tracks.append(tr)


def analyze(video: Path, detector: Detector, prompts: list[str], tracker: Tracker | None = None) -> BlurTrack:
    """Örnek karede tespit; `tracker` verilirse tespitler sonraki örneğe kadar her karede maskeyle takip edilir."""
    cfg = config.load()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    step = max(1, round(fps / cfg["lookahead"]["sample_fps"]))
    gap = cfg["track"]["max_gap_s"]
    bt = BlurTrack(video.name, duration)
    i = 0
    while True:
        chunk: list[np.ndarray] = []
        for _ in range(step):
            ok, frame = cap.read()
            if not ok:
                break
            chunk.append(frame)
        if not chunk:
            break
        t = i / fps
        dets = detector(chunk[0], prompts)
        if tracker is None or not dets:
            associate(bt.tracks, t, dets, gap)
        else:
            small = [_downscale(f) for f in chunk]
            per = tracker.track_objects(small, [d.box for d in dets])
            for j, masks in enumerate(per):
                fdets = [
                    Detection(d.label, b, d.score)
                    for d, m in zip(dets, masks, strict=False)
                    if (b := mask_to_box(m)) is not None
                ]
                associate(bt.tracks, (i + j) / fps, fdets, gap)
        i += len(chunk)
        bt.analyzed_until_s = i / fps
    cap.release()
    bt.analyzed_until_s = duration
    return bt


def _downscale(frame: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Takip küçük karede yapılır (kutular normalize olduğundan sonuç değişmez)."""
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return frame if s >= 1 else cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def render(video: Path, bt: BlurTrack, out: Path) -> None:
    cfg = config.load()["blur"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = bt.active(i / fps, cfg["ramp_in_s"], cfg["ramp_out_s"])
        mask = blur.box_mask((h, w), boxes, int(cfg["dilate_px"]), int(cfg["feather_px"]))
        writer.write(blur.apply(frame, mask, cfg["sigma_px"]))
        i += 1
    cap.release()
    writer.release()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sensa_ai.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("video", type=Path)
    a.add_argument("--proxy", action="store_true", help="kansız vekil dedektör (D-005)")
    a.add_argument("--prompts", default="positive", choices=["positive", "proxy"], help="gerçek modelde istem seti")
    a.add_argument("--no-track", action="store_true", help="maske takibini kapat (daha hızlı, daha kaba)")
    a.add_argument("--device", default=None)
    r = sub.add_parser("render")
    r.add_argument("video", type=Path)
    r.add_argument("track", type=Path)
    r.add_argument("out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        if args.proxy:
            bt = analyze(args.video, ColorProxyDetector(), config.load_prompts()["proxy"])
        else:
            from . import models

            tracker = None if args.no_track else Tracker.from_pretrained(device=args.device)
            # vekil istemle ("red balloon") kapı kapalı: kapının negatifleri kan için ayarlı
            det = models.build_default(device=args.device, use_gate=args.prompts == "positive")
            bt = analyze(args.video, det, config.load_prompts()[args.prompts], tracker)
        out = args.video.with_suffix(".blurtrack.json")
        bt.save(out)
        print(f"{len(bt.tracks)} iz -> {out}")
    else:
        render(args.video, BlurTrack.load(args.track), args.out)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
