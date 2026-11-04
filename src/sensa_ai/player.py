"""Canlı oynatıcı (M3): oynatma kafasının 30 sn önünü tarayan analiz iş parçacığı + bölgesel bulanıklık.

    python -m sensa_ai.player film.mp4 --proxy        # modelsiz vekil (kırmızı alanlar)
    python -m sensa_ai.player film.mp4                # gerçek modeller (M5'te MPS)
    python -m sensa_ai.player film.mp4 --track film.blurtrack.json   # önceden üretilmiş iz

Tuşlar: boşluk = duraklat, ← / → = 10 sn geri / ileri, q = çık.
Ses: PyAV + sounddevice varsa ses ana saattir (görüntü sese uyar, kayma birikmez; audio.py).
Yoksa ffplay ayrı süreçle çalınır (kayma ~100 ms olabilir, ölçülmedi), o da yoksa sessiz.

Güvenlik: analiz yetişmezse (tampon < safe_margin_s) kare tam bulanık gösterilir (D-007).
Bu yüzden ileri sarmada ekran bir süre bulanık kalır; bu bilinçli.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import blur, config
from . import lookahead as la
from .audio import WallClock, make_clock
from .blurtrack import BlurTrack
from .detect import ColorProxyDetector, Detection, Detector
from .pipeline import associate, process_chunk
from .tracking import Tracker


class Analyzer(threading.Thread):
    """Oynatma kafasının önünde koşar; sonuçları ortak BlurTrack'e yazar."""

    def __init__(
        self, video: Path, detector: Detector, prompts: list[str], track: BlurTrack, tracker: Tracker | None = None
    ) -> None:
        super().__init__(daemon=True)
        p = config.load()
        self.video, self.detector, self.prompts, self.track = video, detector, prompts, track
        self.tracker = tracker
        self.window_s = float(p["lookahead"]["window_s"])
        self.sample_fps = float(p["lookahead"]["sample_fps"])
        self.max_gap_s = float(p["track"]["max_gap_s"])
        self.lock = threading.Lock()
        self.playhead_s = 0.0
        self.stop_event = threading.Event()
        self.frames_analyzed = 0

    def set_playhead(self, t: float) -> None:
        with self.lock:
            if t > self.track.analyzed_until_s + 1e-6 or t < self.playhead_s - 1.0:
                # ileri ya da geri arama: analiz o noktadan yeniden başlar
                self.track.analyzed_until_s = t
            self.playhead_s = t

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = 1.0 / self.sample_fps
        n_chunk = max(1, round(fps * step)) if self.tracker else 1
        while not self.stop_event.is_set():
            with self.lock:
                start, end = la.next_window(
                    self.track.analyzed_until_s, self.track.duration_s, self.window_s, self.playhead_s
                )
            if start >= end - 1e-6:
                time.sleep(0.02)
                continue
            t = start
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            chunk = []
            for _ in range(n_chunk):
                ok, frame = cap.read()
                if not ok:
                    break
                chunk.append(frame)
            if not chunk:
                with self.lock:
                    self.track.analyzed_until_s = self.track.duration_s
                continue
            new: list = []
            process_chunk(chunk, t, fps, self.detector, self.prompts, self.tracker, new, self.max_gap_s)
            with self.lock:
                for tr in new:  # aralığın izlerini ortak ize bağla
                    for k in tr.keyframes:
                        associate(self.track.tracks, k.t, [Detection(tr.label, k.box, k.score, k.poly)], self.max_gap_s)
                # arama sırasında playhead değiştiyse eski sonucu ileri taşıma
                if abs(self.track.analyzed_until_s - start) < 1e-6:
                    self.track.analyzed_until_s = min(self.track.duration_s, t + step)
            self.frames_analyzed += 1
        cap.release()


class Compositor:
    """Bir karede ne gösterileceğine karar verir (saf mantık, test edilebilir)."""

    def __init__(self, track: BlurTrack) -> None:
        p = config.load()
        self.track = track
        self.b = p["blur"]
        self.safe_margin_s = float(p["lookahead"]["safe_margin_s"])

    def mode(self, t: float) -> la.Mode:
        return la.mode(t, self.track.analyzed_until_s, self.track.duration_s, self.safe_margin_s)

    def render(self, frame: np.ndarray, t: float) -> tuple[np.ndarray, la.Mode]:
        m = self.mode(t)
        if m is la.Mode.FAILSAFE:
            return blur.apply_full(frame, self.b["failsafe_sigma_px"]), m
        regions = self.track.active_regions(t, **blur.timing(self.b))
        mask = blur.region_mask(frame.shape[:2], regions, int(self.b["dilate_px"]), int(self.b["feather_px"]))
        return blur.apply(frame, mask, self.b["sigma_px"]), m


class Audio:
    """ffplay ile ses (geçici). ffplay yoksa sessiz oynatır."""

    def __init__(self, video: Path) -> None:
        self.video = video
        self.proc: subprocess.Popen[bytes] | None = None
        self.bin = shutil.which("ffplay")

    def start(self, t: float) -> None:
        self.stop()
        if self.bin:
            self.proc = subprocess.Popen(  # noqa: S603
                [self.bin, "-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", f"{t:.2f}", str(self.video)]
            )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


def play(
    video: Path,
    detector: Detector | None,
    prompts: list[str],
    track_file: Path | None = None,
    tracker: Tracker | None = None,
    use_av_audio: bool = True,
) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    track = BlurTrack.load(track_file) if track_file else BlurTrack(video.name, duration)
    analyzer = Analyzer(video, detector, prompts, track, tracker) if detector else None
    comp = Compositor(track)
    if analyzer:
        analyzer.start()
        # açılışta tamponun dolmasını bekle (en çok safe_margin_s kadar analiz)
        t_wait = time.perf_counter()
        while comp.mode(0.0) is la.Mode.FAILSAFE and time.perf_counter() - t_wait < 60:
            time.sleep(0.05)
    clock = make_clock(video) if use_av_audio else None
    audio = None if clock else Audio(video)
    if clock is None:
        clock = WallClock()
        audio.start(0.0)  # type: ignore[union-attr]
    clock.start(0.0)
    paused, pos = False, 0.0
    win = "sensa-ai"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        if not paused:
            pos = clock.now()
        if pos >= duration:
            break
        if analyzer:
            analyzer.set_playhead(pos)
        cap.set(cv2.CAP_PROP_POS_MSEC, pos * 1000) if abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - pos) > 0.5 else None
        ok, frame = cap.read()
        if not ok:
            break
        out, m = comp.render(frame, pos)
        if m is la.Mode.FAILSAFE:
            cv2.putText(out, "analiz yetisiyor...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win, out)
        key = cv2.waitKey(max(1, int(1000 / fps) - 2)) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            if paused:
                clock.pause()
                audio.stop() if audio else None
            else:
                clock.start(pos)
                audio.start(pos) if audio else None
        if key in (81, 2, ord("j")) or key in (83, 3, ord("l")):  # ← / → (platforma göre kodlar), j / l
            pos = max(0.0, min(duration - 0.1, pos + (-10 if key in (81, 2, ord("j")) else 10)))
            cap.set(cv2.CAP_PROP_POS_MSEC, pos * 1000)
            clock.start(pos)
            audio.start(pos) if audio else None
    clock.close()
    audio.stop() if audio else None
    if analyzer:
        analyzer.stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sensa_ai.player")
    ap.add_argument("video", type=Path)
    ap.add_argument("--proxy", action="store_true", help="modelsiz vekil dedektör (D-005)")
    ap.add_argument("--track", type=Path, default=None, help="hazır .blurtrack.json (analiz yok)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-track", action="store_true", help="maske takibini kapat (daha hızlı)")
    ap.add_argument("--profile", default=None, help="hassasiyet: hafif / orta / guclu (params.yaml)")
    args = ap.parse_args(argv)
    config.set_profile(args.profile)
    prompts = config.load_prompts()
    if args.track:
        play(args.video, None, [], args.track)
    elif args.proxy:
        play(args.video, ColorProxyDetector(), prompts["proxy"])
    else:
        from . import models

        tracker = None if args.no_track else Tracker.from_pretrained(device=args.device)
        play(args.video, models.build_default(device=args.device), prompts["positive"], tracker=tracker)


if __name__ == "__main__":
    main()
