"""Uçtan uca vekil test (D-005): kan yok, kırmızı kare var."""

from pathlib import Path

import cv2
import numpy as np

from sensa_ai import pipeline
from sensa_ai.detect import ColorProxyDetector


def test_proxy_end_to_end(proxy_video: Path, tmp_path: Path) -> None:
    bt = pipeline.analyze(proxy_video, ColorProxyDetector(), ["red balloon"])
    assert len(bt.tracks) == 1
    tr = bt.tracks[0]
    assert 0.9 <= tr.start <= 1.1 and 2.4 <= tr.end <= 3.0

    out = tmp_path / "out.mp4"
    pipeline.render(proxy_video, bt, out)
    cap_in, cap_out = cv2.VideoCapture(str(proxy_video)), cv2.VideoCapture(str(out))
    diffs = []
    while True:
        ok1, a = cap_in.read()
        ok2, b = cap_out.read()
        if not (ok1 and ok2):
            break
        # mp4v yeniden kodlaması her kareyi biraz değiştirir; bu yüzden bölge içi farkı
        # bölgeden uzak köşenin farkıyla (sıkıştırma tabanı) karşılaştırıyoruz.
        d = np.abs(a.astype(int) - b.astype(int))
        diffs.append((float(d[40:70, 20:130].mean()), float(d[95:, :20].mean())))
    assert len(diffs) == 40
    early = max(region for region, _ in diffs[:5])  # ilk 0.5 sn: bölge yok
    during = min(region for region, _ in diffs[12:25])  # kare görünürken
    corner = max(c for _, c in diffs)
    assert during > 2 * early  # bölge görünürken belirgin biçimde bulanık
    assert during > 2 * corner  # bulanıklık bölgede, köşede değil


class _BoxTracker:
    """Takip yerine: verilen kutuyu her karede bir piksel sağa kaydırıp maske döndürür."""

    def track_objects(self, frames, boxes, start_idx=0):  # type: ignore[no-untyped-def]
        out = []
        for j, f in enumerate(frames):
            h, w = f.shape[:2]
            ms = []
            for x0, y0, x1, y1 in boxes:
                m = np.zeros((h, w), bool)
                m[int(y0 * h) : int(y1 * h), int(x0 * w) + j : int(x1 * w) + j] = True
                ms.append(m)
            out.append(ms)
        return out


def test_tracker_fills_between_samples(proxy_video: Path) -> None:
    bt = pipeline.analyze(proxy_video, ColorProxyDetector(), ["red balloon"], tracker=_BoxTracker())  # type: ignore[arg-type]
    assert len(bt.tracks) == 1
    # 10 fps video, takip track.fps (6) -> her 2. karede anahtar kare (2 sn x 5 = ~10)
    assert len(bt.tracks[0].keyframes) >= 9
    assert all(k.poly for k in bt.tracks[0].keyframes)  # maske -> poligon izde


def test_make_proxy_video_and_analyze(tmp_path: Path) -> None:
    from tools.make_proxy_video import make

    v = make(tmp_path / "balon.mp4", seconds=16, fps=10, w=320, h=180)
    bt = pipeline.analyze(v, ColorProxyDetector(), ["red balloon"])
    starts = sorted(tr.start for tr in bt.tracks)
    assert starts and 3.5 <= starts[0] <= 4.5  # balon 4. saniyede
    assert all(tr.end <= 14.5 for tr in bt.tracks)


def test_render_keeps_audio(proxy_video: Path, tmp_path: Path) -> None:
    import shutil
    import subprocess

    import pytest

    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg yok")
    with_audio = tmp_path / "sesli.mp4"
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-i", str(proxy_video), "-f", "lavfi", "-i", "sine=frequency=440:duration=4"]
        + ["-c:v", "copy", "-c:a", "aac", "-shortest", str(with_audio)],
        check=True,
    )
    bt = pipeline.analyze(with_audio, ColorProxyDetector(), ["red balloon"])
    out = tmp_path / "out.mp4"
    pipeline.render(with_audio, bt, out)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "video" in probe and "audio" in probe


def test_eval_video_on_proxy(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import json
    import sys

    from tools import eval_video
    from tools.make_proxy_video import make

    vids, labs = tmp_path / "v", tmp_path / "l"
    make(vids / "balon.mp4", seconds=16, fps=10, w=320, h=180)
    labs.mkdir()
    (labs / "balon.json").write_text(json.dumps({"video": "balon.mp4", "events": [{"start": 4.0, "end": 14.0}]}))
    monkeypatch.setattr(eval_video, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", str(vids), "--labels", str(labs), "--proxy"])
    eval_video.main()
    rows = json.loads(next((tmp_path / "experiments" / "results").glob("*_eval_video.json")).read_text())
    assert rows[0]["olay_yakalama"] == 1.0
    assert rows[0]["kare_kapsama_ort"] > 0.85  # 9-10 sn arası balon saklı: kapsama tam olmayabilir


def test_before_after_refuses_positive_and_renders_proxy(tmp_path: Path) -> None:
    import pytest

    from tools import before_after
    from tools.make_proxy_video import make

    with pytest.raises(PermissionError):
        before_after.guard(tmp_path / "pozitif" / "x.mp4", data_root=tmp_path)
    v = make(tmp_path / "balon.mp4", seconds=6, fps=10, w=160, h=90)
    bt = pipeline.analyze(v, ColorProxyDetector(), ["red balloon"])
    out = before_after.make(v, tmp_path / "ba.mp4", bt, vertical=True)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 180 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160


def test_label_events_guard_and_save(tmp_path: Path) -> None:
    import json

    from tools import label_events

    assert label_events.needs_partner(tmp_path / "pozitif" / "a" / "x.mp4", data_root=tmp_path)
    assert not label_events.needs_partner(tmp_path / "vekil" / "x.mp4", data_root=tmp_path)
    out = tmp_path / "e.json"
    label_events.save(out, "x.mp4", [{"start": 1.0, "end": 2.0, "box": None}, {"start": 5.0, "end": None}])
    assert json.loads(out.read_text())["events"] == [{"start": 1.0, "end": 2.0, "box": None}]


def test_tracking_runs_at_track_fps() -> None:
    """Takip her karede değil track.fps hızında: 25 fps, 12 karelik aralık, track_fps 5 -> 3 kare."""
    from sensa_ai.detect import Detection
    from sensa_ai.pipeline import process_chunk

    class StubDet:
        def __call__(self, frame, prompts):  # type: ignore[no-untyped-def]
            return [Detection("x", (0.1, 0.1, 0.3, 0.3), 0.9)]

    class StubTracker:
        seen = 0

        def track_objects(self, frames, boxes):  # type: ignore[no-untyped-def]
            StubTracker.seen = len(frames)
            m = np.zeros(frames[0].shape[:2], bool)
            m[10:30, 10:30] = True
            return [[m] for _ in frames]

    frames = [np.zeros((120, 160, 3), np.uint8) for _ in range(12)]
    tracks: list = []
    process_chunk(frames, 0.0, 25.0, StubDet(), ["x"], StubTracker(), tracks, 1.0, track_fps=5.0)  # type: ignore[arg-type]
    assert StubTracker.seen == 3
    assert [round(k.t, 2) for k in tracks[0].keyframes] == [0.0, 0.2, 0.4]
