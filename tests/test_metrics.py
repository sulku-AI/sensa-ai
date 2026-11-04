import numpy as np
import pytest

from sensa_ai import metrics


def test_polygon_mask_area() -> None:
    m = metrics.polygons_to_mask((100, 100), [[10, 10, 50, 10, 50, 50, 10, 50]])
    assert 1500 < m.sum() < 1800


def test_coverage_and_overblur() -> None:
    gt = metrics.boxes_to_mask((100, 100), [(0.1, 0.1, 0.3, 0.3)])
    pred = metrics.boxes_to_mask((100, 100), [(0.1, 0.1, 0.2, 0.3), (0.8, 0.8, 0.9, 0.9)])
    s = metrics.Score()
    s.add_positive(gt, pred)
    s.add_negative(np.zeros((100, 100), bool))
    s.add_negative(pred)
    out = s.summary()
    assert out["ortalama_kapsama"] == pytest.approx(0.5)
    assert out["yakalama_orani"] == 1.0
    assert out["fazla_bulaniklik_ort"] == pytest.approx(0.01)
    assert out["negatif_yanlis_alarm_orani"] == 0.5


def test_event_scores() -> None:
    from sensa_ai.blurtrack import BlurTrack, Keyframe, Track

    tr = Track("blood", [Keyframe(10.5, (0.1, 0.1, 0.3, 0.3)), Keyframe(15.0, (0.1, 0.1, 0.3, 0.3))])
    tr2 = Track("blood", [Keyframe(50.0, (0.6, 0.6, 0.8, 0.8)), Keyframe(51.0, (0.6, 0.6, 0.8, 0.8))])  # yanlış alarm
    bt = BlurTrack("a.mp4", 120.0, 120.0, [tr, tr2])
    ev = [metrics.Event(10.0, 15.0, (0.0, 0.0, 0.5, 0.5)), metrics.Event(80.0, 82.0)]
    s = metrics.event_scores(bt, ev)
    assert s["olay"] == 2 and s["olay_yakalama"] == 0.5  # ikinci olay kaçtı
    assert 0.85 <= metrics.event_scores(bt, ev[:1])["kare_kapsama_ort"] <= 0.95  # 10.0-10.5 arası açık
    assert s["gec_baslama_ort_s"] == pytest.approx((0.5 + 2.0) / 2, abs=0.11)
    assert s["olay_disi_bulanik_s"] == pytest.approx(1.1, abs=0.15)
    # rampa geç başlamayı kapatır
    assert metrics.event_scores(bt, ev[:1], ramp_in_s=0.6)["gec_baslama_ort_s"] <= 0.01
