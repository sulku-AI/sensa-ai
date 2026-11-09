"""Çevrimdışı boru hattı (M1): video -> .blurtrack.json -> bulanık video.

    python -m sensa_ai.pipeline analyze video.mp4 --proxy
    python -m sensa_ai.pipeline render video.mp4 video.blurtrack.json cikti.mp4

Not: render şimdilik sesi kopyalamaz (OpenCV). Ses ffmpeg ile M1 içinde eklenecek.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import blur, config
from .blurtrack import BlurTrack, Box, Keyframe, Track
from .detect import ColorProxyDetector, Detection, Detector
from .tracking import Tracker, mask_to_box, mask_to_polys


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def associate(tracks: list[Track], t: float, dets: list[Detection], max_gap_s: float, min_iou: float = 0.1) -> None:
    """Açgözlü eşleme: her tespit en çok örtüşen açık ize eklenir, yoksa yeni iz açar.

    Takip açıkken tespitler zaten kare kare maskeden gelir; eşleme yalnız onları izlere bağlar.
    """
    open_tracks = [tr for tr in tracks if t - tr.end <= max_gap_s]
    for d in dets:
        best = max(open_tracks, key=lambda tr: iou(tr.keyframes[-1].box, d.box), default=None)
        if best is not None and iou(best.keyframes[-1].box, d.box) >= min_iou and best.end < t:
            best.keyframes.append(Keyframe(t, d.box, d.score, d.poly))
        else:
            tr = Track(d.label, [Keyframe(t, d.box, d.score, d.poly)])
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
        n_read = 0
        for k in range(step):
            if tracker is None and k > 0:
                # takipsiz hatta aradaki kareler yalnız atlanır: grab() renk çevirmez, kopyalamaz
                ok = cap.grab()
            else:
                ok, frame = cap.read()
                if ok:
                    chunk.append(frame)
            if not ok:
                break
            n_read += 1
        if not chunk:
            break
        process_chunk(chunk, i / fps, fps, detector, prompts, tracker, bt.tracks, gap)
        i += n_read
        bt.analyzed_until_s = i / fps
    cap.release()
    bt.analyzed_until_s = duration
    return bt


def process_chunk(
    chunk: list[np.ndarray],
    t0: float,
    fps: float,
    detector: Detector,
    prompts: list[str],
    tracker: Tracker | None,
    tracks: list[Track],
    gap: float,
    track_fps: float | None = None,
) -> None:
    """Bir örnekleme aralığı: ilk karede tespit, varsa kalan karelerde maske takibi. Çevrimdışı ve oynatıcı ortak.

    Takip her karede değil, `track.fps` hızında yapılır; aradaki kareleri izin ara değeri doldurur.
    """
    dets = detector(chunk[0], prompts)
    if tracker is None or not dets:
        associate(tracks, t0, dets, gap)
        return
    tfps = track_fps or float(config.load()["track"]["fps"])
    step = max(1, round(fps / tfps))
    small = [_downscale(f) for f in chunk[::step]]
    per = tracker.track_objects(small, [d.box for d in dets])
    for j, masks in enumerate(per):
        fdets = [
            Detection(d.label, b, d.score, mask_to_polys(m))
            for d, m in zip(dets, masks, strict=False)
            if (b := mask_to_box(m)) is not None
        ]
        associate(tracks, t0 + j * step / fps, fdets, gap)


def _downscale(frame: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Takip küçük karede yapılır (kutular normalize olduğundan sonuç değişmez)."""
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return frame if s >= 1 else cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def render(video: Path, bt: BlurTrack, out: Path, audio: bool = True) -> None:
    """Bulanık video yazar. ffmpeg varsa ses kaynaktan kopyalanır ve görüntü H.264'e çevrilir."""
    cfg = config.load()["blur"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ffmpeg = shutil.which("ffmpeg") if audio else None
    silent = out.with_name(out.stem + ".sessiz.mp4") if ffmpeg else out
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        regions = bt.active_regions(i / fps, **blur.timing(cfg))
        mask = blur.region_mask((h, w), regions, int(cfg["dilate_px"]), int(cfg["feather_px"]))
        writer.write(blur.apply(frame, mask, cfg["sigma_px"]))
        i += 1
    cap.release()
    writer.release()
    if ffmpeg:
        mux_audio(ffmpeg, silent, video, out)
        silent.unlink(missing_ok=True)


def mux_audio(ffmpeg: str, silent: Path, source: Path, out: Path) -> None:
    """Bulanık görüntü + kaynağın sesi (varsa). Kaynakta ses yoksa yalnız görüntü."""
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(source)]
    cmd += ["-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"]
    cmd += ["-c:a", "copy", "-shortest", str(out)]
    subprocess.run(cmd, check=True)  # noqa: S603


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sensa_ai.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("video", type=Path)
    a.add_argument("--proxy", action="store_true", help="kansız vekil dedektör (D-005)")
    a.add_argument("--prompts", default="positive", choices=["positive", "proxy"], help="gerçek modelde istem seti")
    a.add_argument("--no-track", action="store_true", help="maske takibini kapat (params track.enabled'ı ezer)")
    a.add_argument("--mask-track", action="store_true", help="maske takibini aç (OWLv2 ile önerilir)")
    a.add_argument("--device", default=None)
    a.add_argument("--detector", default=None, help="models.yaml detect adı (ör. rfdetr_seg) ya da .pth yolu")
    a.add_argument("--no-gate", action="store_true", help="SigLIP kapısını kapat (eğitilmiş modelde önerilen)")
    r = sub.add_parser("render")
    r.add_argument("video", type=Path)
    r.add_argument("track", type=Path)
    r.add_argument("out", type=Path)
    ap.add_argument("--profile", default=None, help="hassasiyet: hafif / orta / guclu (params.yaml)")
    args = ap.parse_args(argv)
    config.set_profile(args.profile)

    if args.cmd == "analyze":
        if args.proxy:
            bt = analyze(args.video, ColorProxyDetector(), config.load_prompts()["proxy"])
        else:
            from . import models
            from .tracking import maybe_tracker

            tracker = maybe_tracker(args.device, False if args.no_track else True if args.mask_track else None)
            # vekil istemle ("red balloon") kapı kapalı: kapının negatifleri kan için ayarlı
            gate = False if (args.prompts != "positive" or args.no_gate) else None
            det = models.build_default(device=args.device, use_gate=gate, detector=args.detector)
            bt = analyze(args.video, det, config.load_prompts()[args.prompts], tracker)
        out = args.video.with_suffix(".blurtrack.json")
        bt.save(out)
        print(f"{len(bt.tracks)} iz -> {out}")
    else:
        render(args.video, BlurTrack.load(args.track), args.out)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
