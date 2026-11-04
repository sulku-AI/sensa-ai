import time

import numpy as np
import pytest

from sensa_ai.audio import AudioBuffer, WallClock


def test_buffer_clock_counts_only_real_samples() -> None:
    b = AudioBuffer(rate=1000, channels=2)
    b.reset(10.0)
    b.push(np.ones((300, 2), np.float32))
    out = b.pull(500)
    assert out[:300].all() and not out[300:].any()
    assert b.now() == pytest.approx(10.3)  # sessizlik saati ilerletmez


def test_buffer_split_chunks() -> None:
    b = AudioBuffer(rate=100, channels=1)
    b.push(np.arange(10, dtype=np.float32).reshape(-1, 1))
    assert b.pull(4)[:, 0].tolist() == [0, 1, 2, 3]
    assert b.pull(4)[:, 0].tolist() == [4, 5, 6, 7]
    assert b.buffered == 2


def test_wall_clock_pause() -> None:
    c = WallClock()
    c.start(5.0)
    time.sleep(0.05)
    c.pause()
    t = c.now()
    time.sleep(0.05)
    assert 5.04 <= t <= 5.5 and c.now() == t


def test_av_clock_decodes_with_fake_device(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Gerçek ses aygıtı yerine sahte sounddevice: çözücü + saat birlikte çalışıyor mu."""
    import shutil
    import subprocess
    import sys
    import types

    pytest.importorskip("av")
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg yok")
    wav = tmp_path / "ton.mp4"
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:a", "aac", str(wav)],
        check=True,
    )

    class FakeStream:
        def __init__(self, **kw):  # type: ignore[no-untyped-def]
            self.cb, self.active = kw["callback"], False

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=FakeStream))
    from sensa_ai.audio import AvAudioClock

    c = AvAudioClock(wav)
    c.start(1.0)
    deadline = time.time() + 5
    while c.buf.buffered < 48000 and time.time() < deadline:
        time.sleep(0.01)
    out = np.zeros((24000, 2), np.float32)
    c.stream.cb(out, 24000, None, None)  # aygıtın yarım saniye çalması
    assert np.abs(out).max() > 0.05  # ses var
    assert c.now() == pytest.approx(1.5, abs=0.01)
    c.close()


def test_silent_video_uses_wall_clock(proxy_video) -> None:  # type: ignore[no-untyped-def]
    """Sessiz videoda ses saati ilerlemezdi (oynatıcı 0-0.5 sn arasında dönüyordu)."""
    from sensa_ai.audio import WallClock, make_clock

    assert isinstance(make_clock(proxy_video), WallClock)
