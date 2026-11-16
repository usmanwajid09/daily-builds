import numpy as np
import pytest

from skin_detector import detect_skin, rgb_to_ycbcr, skin_stats
from skin_detector.demo import RECT, SKIN_COLOR, make_scene

SKIN = np.array(SKIN_COLOR, dtype=np.float64)


def _solid(color, shape=(32, 32)):
    return np.tile(np.asarray(color, dtype=np.float64), (*shape, 1))


def test_solid_skin_colour_is_all_skin():
    mask = detect_skin(_solid(SKIN))
    assert mask.dtype == bool
    assert mask.all()


def test_blue_is_not_skin():
    assert not detect_skin(_solid((20, 40, 120))).any()


def test_green_is_not_skin():
    assert not detect_skin(_solid((30, 160, 60))).any()


def test_ycbcr_conversion_of_known_colour():
    # Pure red (255, 0, 0): Y=76.245, Cb=84.97, Cr=255.5 (BT.601).
    ycbcr = rgb_to_ycbcr(_solid((255, 0, 0), shape=(1, 1)))
    assert ycbcr[0, 0, 0] == pytest.approx(76.245, abs=0.01)
    assert ycbcr[0, 0, 1] == pytest.approx(84.97, abs=0.01)
    assert ycbcr[0, 0, 2] == pytest.approx(255.5, abs=0.01)


def test_ycbcr_of_skin_colour_inside_bounds():
    ycbcr = rgb_to_ycbcr(_solid(SKIN, shape=(1, 1)))
    assert 77 <= ycbcr[0, 0, 1] <= 127
    assert 133 <= ycbcr[0, 0, 2] <= 173


def test_scene_stats_match_known_rectangle():
    stats = skin_stats(make_scene())
    expected_pixels = (RECT[2] - RECT[0] + 1) * (RECT[3] - RECT[1] + 1)
    assert stats["skin_pixels"] == expected_pixels
    assert stats["skin_fraction"] == pytest.approx(
        expected_pixels / (120 * 160)
    )
    assert stats["largest_bbox"] == RECT
    assert stats["largest_pixels"] == expected_pixels


def test_largest_region_picks_bigger_of_two_rects():
    image = _solid((20, 40, 120), shape=(40, 60))
    image[2:6, 2:6] = SKIN          # 4x4 = 16 pixels
    image[20:30, 30:50] = SKIN      # 10x20 = 200 pixels
    stats = skin_stats(image)
    assert stats["skin_pixels"] == 216
    assert stats["largest_bbox"] == (20, 30, 29, 49)
    assert stats["largest_pixels"] == 200


def test_no_skin_returns_none_bbox():
    stats = skin_stats(_solid((20, 40, 120)))
    assert stats["skin_fraction"] == pytest.approx(0.0)
    assert stats["skin_pixels"] == 0
    assert stats["largest_bbox"] is None
    assert stats["largest_pixels"] == 0


def test_invalid_image_shape_raises():
    with pytest.raises(ValueError):
        detect_skin(np.zeros((32, 32)))


def test_out_of_range_values_raise():
    with pytest.raises(ValueError):
        detect_skin(np.full((8, 8, 3), 300.0))


def test_invalid_range_raises():
    with pytest.raises(ValueError):
        detect_skin(_solid(SKIN), cb_range=(127, 77))
    with pytest.raises(ValueError):
        detect_skin(_solid(SKIN), cr_range=(200, 100))
