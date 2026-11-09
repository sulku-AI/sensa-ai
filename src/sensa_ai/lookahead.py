"""İleri bakma tamponu: analiz oynatma kafasının ne kadar önünde?"""

from __future__ import annotations

from enum import Enum


class Mode(Enum):
    NORMAL = "normal"  # tampon yeterli, iz uygulanır
    FAILSAFE = "failsafe"  # analiz yetişmiyor: güvenli mod (D-007)


def buffer_s(playhead_s: float, analyzed_until_s: float, duration_s: float) -> float:
    """Kalan analiz edilmiş süre. Videonun sonu analiz edildiyse sonsuz."""
    if analyzed_until_s >= duration_s:
        return float("inf")
    return analyzed_until_s - playhead_s


def mode(playhead_s: float, analyzed_until_s: float, duration_s: float, safe_margin_s: float) -> Mode:
    return Mode.NORMAL if buffer_s(playhead_s, analyzed_until_s, duration_s) >= safe_margin_s else Mode.FAILSAFE


def next_window(analyzed_until_s: float, duration_s: float, window_s: float, playhead_s: float) -> tuple[float, float]:
    """Analiz iş parçacığının sıradaki işlenecek aralığı: [başlangıç, bitiş)."""
    start = max(analyzed_until_s, playhead_s)
    return start, min(duration_s, playhead_s + window_s)
