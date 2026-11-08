"""2.1 iyileştirmeleri: ayar önbelleği, aramasız analiz, işaret çizelgesi, eğitim gruplaması. Yalnız sentetik."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from sensa_ai import config


def test_params_cache_reloads_after_edit(tmp_path: Path) -> None:
    p = tmp_path / "params.yaml"
    p.write_text('blur:\n  sigma_px: {value: 1.0, status: estimate, source: "x"}\n')
    assert config.load(p)["blur"]["sigma_px"] == 1.0
    a = config.load_raw(p)
    a["blur"]["sigma_px"]["value"] = 99.0  # çağıranın değiştirmesi önbelleği bozmaz
    assert config.load(p)["blur"]["sigma_px"] == 1.0
    p.write_text('blur:\n  sigma_px: {value: 2.0, status: estimate, source: "x"}\n')
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # aynı saniyede yazılsa da yeni sayılsın
    assert config.load(p)["blur"]["sigma_px"] == 2.0


def test_params_cache_is_fast() -> None:
    config.load()
    t0 = time.perf_counter()
    for _ in range(200):
        config.load()
    assert (time.perf_counter() - t0) / 200 < 0.002  # önceden ~6 ms / çağrı (her karede)


class _FakeCap:
    def __init__(self) -> None:
        self.sets: list[float] = []
        self.grabs = 0

    def set(self, prop: int, value: float) -> bool:
        self.sets.append(value)
        return True

    def grab(self) -> bool:
        self.grabs += 1
        return True


def test_analyzer_skips_forward_instead_of_seeking() -> None:
    from sensa_ai.player import Analyzer

    cap = _FakeCap()
    nxt = Analyzer._seek(cap, 24.0, 10.0, -1)  # type: ignore[arg-type]  # ilk kez: arama
    assert cap.sets == [10000.0] and nxt == 240
    nxt = Analyzer._seek(cap, 24.0, 10.0 + 1 / 6, nxt + 1)  # sonraki örnek: 3 kare atla, arama yok
    assert len(cap.sets) == 1 and cap.grabs == 3 and nxt == 244
    Analyzer._seek(cap, 24.0, 60.0, nxt + 1)  # uzak ileri: arama
    Analyzer._seek(cap, 24.0, 5.0, 1441)  # geri: arama
    assert len(cap.sets) == 3


def test_flag_timeline_counts_proxy_and_false_alarms(proxy_video: Path) -> None:
    from sensa_ai.detect import ColorProxyDetector
    from tools import flag_timeline as ft

    times, scores, duration = ft.sample_flags(proxy_video, [ColorProxyDetector()], 2.0, 0.5, ["red"])
    flags = [s > 0 for s in scores[0]]
    assert len(times) == 8 and abs(duration - 4.0) < 1e-6
    assert [t for t, f in zip(times, flags, strict=True) if f] == [1.0, 1.5, 2.0, 2.5]
    # kırmızı kare "gerçek" sayılırsa: içeride hep işaretli, dışarıda yanlış alarm yok
    s = ft.summarize(times, flags, duration, [(1.0, 2.9)], 0.5)
    assert s["ici_isaretli_oran"] == 1.0 and s["disi_yanlis_alarm_olayi"] == 0 and s["yakalanan_aralik"] == "1/1"
    # gerçek aralık başka yerdeyse: art arda işaretler tek yanlış alarm olayı
    s = ft.summarize(times, flags, duration, [(3.2, 3.9)], 0.5)
    assert s["disi_yanlis_alarm_olayi"] == 1 and s["yakalanan_aralik"] == "0/1"
    assert s["dakika"] == [4]


def test_negative_video_frames_stay_in_one_split_and_are_not_capped(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from tools import train_rfdetr as tr

    neg = tmp_path / "video_zor" / "images"
    neg.mkdir(parents=True)
    for v in range(12):
        for f in range(6):
            cv2.imwrite(str(neg / f"ep{v}_{f * 24:07d}.jpg"), np.full((16, 16, 3), 90, np.uint8))
    out = tr.split_items(tr.collect([], [neg.parent]), aug_per_image=2)
    where: dict[str, set[str]] = {}
    for split, rows in out.items():
        for r in rows:
            where.setdefault(r[4], set()).add(split)
    assert len(where) == 12 and all(len(s) == 1 for s in where.values())  # video başına tek bölüm
    assert sum(len(r) for r in out.values()) == 72  # kopya sınırı (2) video karelerine uygulanmaz
