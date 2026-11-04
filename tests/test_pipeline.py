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
