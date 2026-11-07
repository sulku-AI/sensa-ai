from pathlib import Path

import pytest

from sensa_ai.blurtrack import BlurTrack, Keyframe, Track


def make() -> BlurTrack:
    tr = Track("blood", [Keyframe(1.0, (0.0, 0.0, 0.2, 0.2)), Keyframe(2.0, (0.2, 0.0, 0.4, 0.2))])
    return BlurTrack("a.mp4", 10.0, 10.0, [tr])


def test_interpolation() -> None:
    box = make().tracks[0].box_at(1.5)
    assert box == pytest.approx((0.1, 0.0, 0.3, 0.2))


def test_ramp_starts_before_and_ends_after() -> None:
    bt = make()
    assert bt.active(0.5, ramp_in_s=0.3) == []
    ((_, s),) = bt.active(0.85, ramp_in_s=0.3)
    assert 0 < s < 1
    ((_, s),) = bt.active(2.25, ramp_out_s=0.5)
    assert s == pytest.approx(0.5)
    assert bt.active(2.6, ramp_out_s=0.5) == []


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.blurtrack.json"
    make().save(p)
    assert BlurTrack.load(p) == make()


def test_poly_moves_with_box_and_roundtrips(tmp_path: Path) -> None:
    poly = [[0.0, 0.0, 0.2, 0.0, 0.0, 0.2]]
    tr = Track("blood", [Keyframe(1.0, (0.0, 0.0, 0.2, 0.2), 1.0, poly), Keyframe(2.0, (0.2, 0.0, 0.4, 0.2))])
    bt = BlurTrack("a.mp4", 10.0, 10.0, [tr])
    (r,) = bt.active_regions(1.2)
    assert r.poly is not None and r.poly[0][0] == pytest.approx(0.04)  # kutu 0.04 sağa kaydı
    p = tmp_path / "a.blurtrack.json"
    bt.save(p)
    assert BlurTrack.load(p) == bt


def test_hold_covers_sampling_gap_then_ramps() -> None:
    bt = make()  # iz 1.0 - 2.0
    kw = {"ramp_in_s": 0.3, "ramp_out_s": 0.5, "hold_in_s": 0.5, "hold_out_s": 0.5}
    assert bt.active(0.6, **kw)[0][1] == 1.0  # tutma: tam güç
    ((_, s),) = bt.active(0.35, **kw)  # tutmadan önce rampa
    assert 0 < s < 1
    assert bt.active(0.1, **kw) == []
    assert bt.active(2.4, **kw)[0][1] == 1.0
    ((_, s),) = bt.active(2.75, **kw)
    assert s == pytest.approx(0.5)


def test_timing_from_params() -> None:
    from sensa_ai import blur, config

    t = blur.timing(config.load()["blur"])
    assert t["hold_in_s"] == pytest.approx(1 / config.load()["lookahead"]["sample_fps"])
