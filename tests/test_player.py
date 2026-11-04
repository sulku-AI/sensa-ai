"""M3 oynatıcı mantığı: pencere açmadan, vekil video ile."""

import time
from pathlib import Path

import numpy as np

from sensa_ai import lookahead as la
from sensa_ai.blurtrack import BlurTrack, Keyframe, Track
from sensa_ai.detect import ColorProxyDetector
from sensa_ai.player import Analyzer, Compositor

from .conftest import red_square_frame


def test_analyzer_runs_ahead_of_playhead(proxy_video: Path) -> None:
    track = BlurTrack("vekil.mp4", 4.0)
    an = Analyzer(proxy_video, ColorProxyDetector(), ["red balloon"], track)
    an.start()
    deadline = time.time() + 10
    while track.analyzed_until_s < track.duration_s and time.time() < deadline:
        time.sleep(0.02)
    an.stop_event.set()
    assert track.analyzed_until_s >= track.duration_s
    assert len(track.tracks) == 1 and 0.9 <= track.tracks[0].start <= 1.1


def test_seek_forward_restarts_analysis(proxy_video: Path) -> None:
    track = BlurTrack("vekil.mp4", 100.0)  # uzun gibi davran: analiz sona varmasın
    an = Analyzer(proxy_video, ColorProxyDetector(), ["red balloon"], track)
    an.set_playhead(50.0)
    assert track.analyzed_until_s == 50.0


def test_compositor_failsafe_then_normal() -> None:
    tr = Track("blood", [Keyframe(1.0, (0.3, 0.3, 0.6, 0.6)), Keyframe(2.0, (0.3, 0.3, 0.6, 0.6))])
    track = BlurTrack("a.mp4", 60.0, analyzed_until_s=3.0, tracks=[tr])
    comp = Compositor(track)
    frame = red_square_frame()
    _, m = comp.render(frame, 1.5)
    assert m is la.Mode.FAILSAFE  # tampon 1.5 sn < safe_margin_s
    track.analyzed_until_s = 40.0
    out, m = comp.render(frame, 1.5)
    assert m is la.Mode.NORMAL
    assert np.array_equal(out[:5, :5], frame[:5, :5]) and not np.array_equal(out, frame)
