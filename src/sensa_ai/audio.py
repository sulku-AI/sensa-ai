"""Ses ve ana saat (M3): PyAV ile çöz, sounddevice ile çal; oynatıcının zamanı sesin çalınan örnek sayısından gelir.

Görüntü bu saate uyar (ses saati ana saattir), böylece uzun filmde kayma birikmez. sounddevice
ya da ses aygıtı yoksa `WallClock` kullanılır (sessiz, duvar saati).

    pip install -e ".[player]"   # av + sounddevice (macOS'ta PortAudio tekerlekle gelir)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Clock(Protocol):
    def start(self, t: float) -> None: ...
    def pause(self) -> None: ...
    def now(self) -> float: ...
    def close(self) -> None: ...


class WallClock:
    """Sessiz yedek: duvar saati."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._pos = 0.0
        self._running = False

    def start(self, t: float) -> None:
        self._pos, self._t0, self._running = t, time.perf_counter(), True

    def pause(self) -> None:
        self._pos, self._running = self.now(), False

    def now(self) -> float:
        return self._pos + (time.perf_counter() - self._t0 if self._running else 0.0)

    def close(self) -> None:
        self._running = False


class AudioBuffer:
    """Çözücü iş parçacığı yazar, ses geri çağrısı okur. Saat = başlangıç + çalınan örnek / hız."""

    def __init__(self, rate: int, channels: int, max_s: float = 4.0) -> None:
        self.rate, self.channels = rate, channels
        self.max_samples = int(max_s * rate)
        self.chunks: deque[np.ndarray] = deque()
        self.buffered = 0
        self.played = 0
        self.start_s = 0.0
        self.lock = threading.Lock()
        self.eof = False

    def reset(self, start_s: float) -> None:
        with self.lock:
            self.chunks.clear()
            self.buffered = self.played = 0
            self.start_s, self.eof = start_s, False

    def full(self) -> bool:
        return self.buffered >= self.max_samples

    def push(self, pcm: np.ndarray) -> None:
        """pcm: (örnek, kanal) float32."""
        with self.lock:
            self.chunks.append(pcm)
            self.buffered += len(pcm)

    def pull(self, n: int) -> np.ndarray:
        """n örnek döndürür; yetmezse sessizlikle doldurur (sessizlik saati ilerletmez)."""
        out = np.zeros((n, self.channels), np.float32)
        k = 0
        with self.lock:
            while k < n and self.chunks:
                c = self.chunks[0]
                take = min(n - k, len(c))
                out[k : k + take] = c[:take]
                k += take
                if take == len(c):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = c[take:]
            self.buffered -= k
            self.played += k
        return out

    def now(self) -> float:
        return self.start_s + self.played / self.rate


class AvAudioClock:
    """PyAV çözücü + sounddevice çıkışı."""

    def __init__(self, video: Path, rate: int = 48000, channels: int = 2) -> None:
        import av
        import sounddevice as sd

        self.av, self.video = av, video
        self.buf = AudioBuffer(rate, channels)
        self.stream: Any = sd.OutputStream(samplerate=rate, channels=channels, dtype="float32", callback=self._cb)
        self._paused = True
        self._gen = 0
        self._thread: threading.Thread | None = None

    def _cb(self, outdata: np.ndarray, frames: int, _time: Any, _status: Any) -> None:
        outdata[:] = 0 if self._paused else self.buf.pull(frames)

    def _decode(self, gen: int, t: float) -> None:
        av = self.av
        with av.open(str(self.video)) as c:
            if not c.streams.audio:
                self.buf.eof = True
                return
            s = c.streams.audio[0]
            c.seek(int(t / s.time_base), stream=s) if s.time_base else None
            rs = av.AudioResampler(format="flt", layout="stereo", rate=self.buf.rate)
            for frame in c.decode(s):
                if gen != self._gen:
                    return
                if frame.time is not None and frame.time + frame.samples / frame.sample_rate < t:
                    continue  # aramadan önceki parçayı at
                for r in rs.resample(frame):
                    self.buf.push(r.to_ndarray().reshape(-1, self.buf.channels))
                while self.buf.full() and gen == self._gen:
                    time.sleep(0.01)
        self.buf.eof = True

    def start(self, t: float) -> None:
        self._gen += 1
        self.buf.reset(t)
        self._thread = threading.Thread(target=self._decode, args=(self._gen, t), daemon=True)
        self._thread.start()
        self._paused = False
        if not self.stream.active:
            self.stream.start()

    def pause(self) -> None:
        self._paused = True

    def now(self) -> float:
        return self.buf.now()

    def close(self) -> None:
        self._gen += 1
        self.stream.stop()
        self.stream.close()


def make_clock(video: Path) -> Clock:
    try:
        return AvAudioClock(video)
    except Exception:  # noqa: BLE001  (av/sounddevice/aygıt yok)
        return WallClock()
