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
