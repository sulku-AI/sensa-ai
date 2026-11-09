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


def test_judges_run_only_on_flagged_samples_and_suggest_minutes(proxy_video: Path) -> None:
    from sensa_ai.detect import ColorProxyDetector, Detection
    from tools import flag_timeline as ft

    calls = []

    def judge(frame, prompts):  # type: ignore[no-untyped-def]
        calls.append(1)
        return [Detection("kare", (0.0, 0.0, 1.0, 1.0), 0.9)]

    times, scores, _ = ft.sample_flags(proxy_video, [ColorProxyDetector()], 2.0, 0.5, ["red"], [judge])
    assert len(calls) == 4  # yalnız ilk dedektörün işaretlediği 4 anda
    assert [s > 0 for s in scores[1]] == [s > 0 for s in scores[0]]
    assert ft.candidate_ranges([0, 6, 9, 0, 2, 7], 5) == "1-3,5-6"
    assert ft.candidate_ranges([0, 0], 5) == ""


class _Gated:
    """build_default yerine: vekil dedektör, eşiği değiştirilebilir `detector` alanıyla."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        from sensa_ai.detect import ColorProxyDetector

        self.detector = SimpleNamespace(box_threshold=0.3)
        self._det = ColorProxyDetector()

    def __call__(self, frame, prompts):  # type: ignore[no-untyped-def]
        return self._det(frame, ["red"])


def test_autolabel_lowers_detector_threshold_and_takes_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proxy_video: Path
) -> None:
    import json
    import sys

    from sensa_ai import models, safety
    from tools import autolabel

    data = tmp_path / "data"
    monkeypatch.setattr(safety, "DATA_ROOT", data)
    monkeypatch.setattr(autolabel, "ROOT", tmp_path)
    gated = _Gated()
    monkeypatch.setattr(models, "build_default", lambda device=None, detector=None: gated)
    original = safety.write_dataset_frame
    monkeypatch.setattr(safety, "write_dataset_frame", lambda p, f, data_root=data: original(p, f, data_root))
    out = data / "pozitif" / "al"
    argv = ["al", str(proxy_video), "--out", str(out), "--fps", "2", "--min-score", "0.2", "--keep-empty", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    autolabel.main()
    assert gated.detector.box_threshold == 0.2  # önceden 0.3'te kalıyordu: --min-score 0.2 etkisizdi
    coco = json.loads((out / "_annotations.coco.json").read_text())
    assert [im["time_s"] for im in coco["images"]] == [1.0, 1.5, 2.0, 2.5]  # tek dosya da kabul


def test_relabel_unsure_ranges_and_negative_cap() -> None:
    from tools import relabel_events as rl

    # 1 fps, 0..299 s: her saniye tespitli kare (model her yerde "kan" diyor)
    imgs = [{"id": i, "file_name": f"images/ep_{i * 25:07d}.jpg", "time_s": float(i)} for i in range(300)]
    anns = [{"id": i, "image_id": i, "score": 0.9} for i in range(300)]
    coco = {"images": imgs, "annotations": anns}
    out, st = rl.relabel(coco, "ep", 25.0, [(60.0, 120.0)], 0.5, unsure=[(180.0, 240.0)], neg_per_min=12)
    times = [im["time_s"] for im in out["images"]]
    assert st["pozitif_kaldi"] == 65  # 58..122 s (2 sn pay)
    assert st["belirsiz_atildi"] == 65  # 178..242 s (2 sn pay)
    assert not any(178 <= t <= 242 for t in times)  # belirsiz dakikalar eğitimden çıktı
    neg = [t for t in times if not 58 <= t <= 122]
    assert len(neg) == st["negatif_oldu"] and max(neg.count(m) for m in neg) == 1
    per_min = {m: sum(1 for t in neg if int(t // 60) == m) for m in range(5)}
    assert all(n <= 12 for n in per_min.values()) and st["fazla_negatif_atildi"] > 0
    assert {a["image_id"] for a in out["annotations"]} == {int(t) for t in times if 58 <= t <= 122}
    # sınırsız (0): bütün aralık dışı kareler negatif kalır
    _, st0 = rl.relabel(coco, "ep", 25.0, [(60.0, 120.0)], 0.5, unsure=[(180.0, 240.0)], neg_per_min=0)
    assert st0["negatif_oldu"] == 300 - 65 - st0["belirsiz_atildi"] and st0["fazla_negatif_atildi"] == 0


def test_flag_timeline_unsure_suggestions_and_safe_names(tmp_path: Path) -> None:
    from tools import flag_timeline as ft

    times = [float(t) for t in range(10)]
    flags = [t in (1.0, 2.0, 5.0, 6.0, 8.0) for t in times]
    s = ft.summarize(times, flags, 10.0, [(1.0, 2.0)], 1.0, unsure=[(5.0, 6.0)])
    assert s["belirsiz_ornek"] == 2 and s["disi_yanlis_alarm_olayi"] == 1  # yalnız 8. sn sayılır
    minutes = [0] * 51
    for m, n in {5: 5, 16: 2, 27: 5, 28: 2, 29: 3, 38: 3, 49: 1}.items():
        minutes[m] = n
    assert ft.candidate_ranges(minutes, 5) == "5-6,27-28"
    assert ft.unsure_ranges(minutes, 5, 2, 1) == "4-5,6-7,16-17,26-27,28-30,38-39"
    assert ft.short_name("rfdetr:models/rfdetr_v2b/checkpoint_best_ema.pth") == "rfdetr_v2b/best_ema"
    assert ft.short_name("owlv2_base") == "owlv2_base"
    video = tmp_path / "ep.mp4"
    first = ft.result_stem(tmp_path, video, 0.3, True)
    (tmp_path / f"{first}.md").write_text("x")
    second = ft.result_stem(tmp_path, video, 0.3, True)
    assert first.endswith("_isaret_ep_e0.3_hakem") and second == first + "_2"


def test_prepare_refuses_unrelabeled_series_frames(tmp_path: Path) -> None:
    import argparse
    import json

    from tools import train_rfdetr as tr

    al = tmp_path / "autolabel"
    al.mkdir()
    coco = {"images": [{"id": 1, "file_name": "images/dex1-1_0000024.jpg", "width": 4, "height": 4}]}
    (al / "_annotations.coco.json").write_text(json.dumps({**coco, "annotations": []}))
    assert tr.unrelabeled_videos(al) == ["dex1-1"]
    args = argparse.Namespace(src=[], neg=[], train_only=[al], aug_per_image=2)
    with pytest.raises(SystemExit, match="relabel_events"):
        tr.prepare(args)
    (al / "_annotations.coco.json").write_text(json.dumps({**coco, "annotations": [], "relabel": {"dex1-1": {}}}))
    assert tr.unrelabeled_videos(al) == []
