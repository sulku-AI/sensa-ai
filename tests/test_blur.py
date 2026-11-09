import numpy as np

from sensa_ai import blur

from .conftest import red_square_frame


def test_outside_mask_unchanged_inside_blurred() -> None:
    frame = red_square_frame()
    boxes = [((0.3, 0.3, 0.6, 0.6), 1.0)]
    mask = blur.box_mask(frame.shape[:2], boxes, dilate_px=0, feather_px=4)
    out = blur.apply(frame, mask, sigma_px=6)
    assert np.array_equal(out[:10, :10], frame[:10, :10])
    inside = (slice(45, 65), slice(55, 85))
    assert out[inside].std() < frame[inside].std()


def test_strength_scales_mask() -> None:
    m = blur.box_mask((100, 100), [((0.2, 0.2, 0.8, 0.8), 0.5)], dilate_px=0, feather_px=0)
    assert np.isclose(m.max(), 0.5)


def test_empty_mask_returns_same_frame() -> None:
    frame = red_square_frame()
    assert blur.apply(frame, np.zeros(frame.shape[:2], np.float32), 10) is frame


def test_polygon_region_follows_shape() -> None:
    from sensa_ai.blurtrack import Region

    tri = [[0.1, 0.1, 0.9, 0.1, 0.1, 0.9]]  # sol üst üçgen
    m = blur.region_mask((100, 100), [Region((0.1, 0.1, 0.9, 0.9), 1.0, tri)], dilate_px=0, feather_px=0)
    assert m[20, 20] == 1.0 and m[85, 85] == 0.0  # kutunun içinde ama üçgenin dışında


def test_region_without_poly_uses_box() -> None:
    from sensa_ai.blurtrack import Region

    m = blur.region_mask((100, 100), [Region((0.2, 0.2, 0.4, 0.4), 0.5)], dilate_px=0, feather_px=0)
    assert np.isclose(m[30, 30], 0.5) and m[80, 80] == 0
