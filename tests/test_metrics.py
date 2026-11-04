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
