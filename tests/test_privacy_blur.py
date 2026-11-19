import numpy as np
import pytest

from privacy_blur.demo import RECT, SKIN_COLOR, make_scene
from privacy_blur.privacy import (
    blur_region,
    blur_stats,
    detect_skin,
    privacy_blur,
    rgb_to_ycbcr,
)

SKIN = (210, 170, 140)


def test_solid_skin_image_all_masked():
    image = np.zeros((10, 10, 3), dtype=np.float64)
    image[:, :] = SKIN
    mask = detect_skin(image)
    assert mask.dtype == bool
    assert mask.shape == (10, 10)
    assert mask.all()


def test_blue_is_not_skin():
    image = np.zeros((8, 8, 3), dtype=np.float64)
    image[:, :] = (20, 40, 200)  # pure-ish blue
    assert not detect_skin(image).any()


def test_green_is_not_skin():
    image = np.zeros((8, 8, 3), dtype=np.float64)
    image[:, :] = (30, 160, 60)  # green
    assert not detect_skin(image).any()


def test_ycbcr_of_pure_red():
    red = np.array([[[255.0, 0.0, 0.0]]])
    ycbcr = rgb_to_ycbcr(red)
    assert ycbcr.shape == (1, 1, 3)
    assert np.allclose(ycbcr[0, 0], (76.245, 84.97, 255.5), atol=0.01)


def test_blur_changes_skin_but_leaves_non_skin_identical():
    image = make_scene()
    blurred, mask = privacy_blur(image)
    assert blurred.shape == image.shape
    # Some pixels inside the rectangle must change (edges mix with
    # the background); the uniform interior may stay the same.
    r_min, c_min, r_max, c_max = RECT
    rect = (slice(r_min, r_max + 1), slice(c_min, c_max + 1))
    assert np.any(blurred[rect] != image[rect])
    # Everything outside the skin mask is pixel-identical.
    assert np.array_equal(blurred[~mask], image[~mask])


def test_kernel_size_one_leaves_image_unchanged():
    image = make_scene()
    blurred, mask = privacy_blur(image, kernel_size=1)
    assert mask.sum() > 0
    assert np.array_equal(blurred, image)


def test_wrong_ndim_raises():
    with pytest.raises(ValueError):
        detect_skin(np.zeros((10, 10)))  # 2-D, no channels


def test_empty_image_raises():
    with pytest.raises(ValueError):
        detect_skin(np.zeros((0, 0, 3)))


def test_out_of_range_values_raise():
    too_high = np.zeros((4, 4, 3), dtype=np.float64)
    too_high[0, 0, 0] = 256.0
    with pytest.raises(ValueError):
        detect_skin(too_high)
    too_low = np.zeros((4, 4, 3), dtype=np.float64)
    too_low[0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        detect_skin(too_low)


def test_even_kernel_size_raises():
    image = np.zeros((6, 6, 3), dtype=np.float64)
    image[:, :] = SKIN
    mask = detect_skin(image)
    with pytest.raises(ValueError):
        blur_region(image, mask, kernel_size=4)


def test_bad_ranges_raise():
    image = np.zeros((4, 4, 3), dtype=np.float64)
    image[:, :] = SKIN
    with pytest.raises(ValueError):
        detect_skin(image, cb_range=(127, 77))  # reversed
    with pytest.raises(ValueError):
        detect_skin(image, cr_range=(173,))  # not a pair


def test_scene_stats_match_known_rectangle():
    image = make_scene()
    stats = blur_stats(image)
    rect_pixels = (RECT[2] - RECT[0] + 1) * (RECT[3] - RECT[1] + 1)
    assert stats["skin_pixels"] == rect_pixels == 3000
    assert stats["skin_fraction"] == pytest.approx(
        rect_pixels / (120 * 160)
    )
    # Only edge pixels of the uniform rectangle change value, so the
    # changed count is positive but smaller than the skin count.
    assert 0 < stats["blurred_pixels"] < stats["skin_pixels"]
