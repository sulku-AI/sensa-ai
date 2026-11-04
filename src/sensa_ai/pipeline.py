"""Çevrimdışı boru hattı (M1): video -> .blurtrack.json -> bulanık video.

    python -m sensa_ai.pipeline analyze video.mp4 --proxy
    python -m sensa_ai.pipeline render video.mp4 video.blurtrack.json cikti.mp4

Not: render şimdilik sesi kopyalamaz (OpenCV). Ses ffmpeg ile M1 içinde eklenecek.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from . import blur, config
from .blurtrack import BlurTrack, Box, Keyframe, Track
from .detect import ColorProxyDetector, Detection, Detector


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


def analyze(video: Path, detector: Detector, prompts: list[str]) -> BlurTrack:
    cfg = config.load()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    step = max(1, round(fps / cfg["lookahead"]["sample_fps"]))
    bt = BlurTrack(video.name, duration)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            associate(bt.tracks, t, detector(frame, prompts), cfg["track"]["max_gap_s"])
            bt.analyzed_until_s = t
        i += 1
    cap.release()
    bt.analyzed_until_s = duration
    return bt


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
    r = sub.add_parser("render")
    r.add_argument("video", type=Path)
    r.add_argument("track", type=Path)
    r.add_argument("out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        if not args.proxy:
            raise SystemExit("Gerçek modeller M1'de eklenecek; şimdilik --proxy kullan.")
        prompts = config.load_prompts()["proxy"]
        bt = analyze(args.video, ColorProxyDetector(), prompts)
        out = args.video.with_suffix(".blurtrack.json")
        bt.save(out)
        print(f"{len(bt.tracks)} iz -> {out}")
    else:
        render(args.video, BlurTrack.load(args.track), args.out)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
