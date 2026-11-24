import numpy as np
import pytest

from histogram_backprojection.backprojection import (
    backproject,
    build_histogram,
    localise,
    locate_centroid,
    locate_peak,
)
from histogram_backprojection.demo import make_test_image


def _histogram(image=None, roi=(54, 24, 12, 12)):
    image = image if image is not None else make_test_image()
    return build_histogram(image, roi=roi, bins=16)


# --- build_histogram ------------------------------------------------------


def test_histogram_sums_to_one_and_has_shape():
    hist = _histogram()
    assert hist.shape == (16, 16, 16)
    assert hist.sum() == pytest.approx(1.0)


def test_histogram_of_pure_roi_is_single_bin():
    hist = _histogram()
    assert hist.max() == pytest.approx(1.0)
    assert np.count_nonzero(hist) == 1


def test_histogram_of_uniform_image_is_single_bin():
    image = np.full((10, 10, 3), [10.0, 200.0, 90.0])
    hist = build_histogram(image, roi=(0, 0, 10, 10), bins=8)
    assert hist.sum() == pytest.approx(1.0)
    assert np.count_nonzero(hist) == 1


def test_histogram_invalid_roi_and_bins_raise():
    image = make_test_image()
    with pytest.raises(ValueError, match="outside"):
        build_histogram(image, roi=(95, 75, 10, 10), bins=16)
    with pytest.raises(ValueError, match="positive"):
        build_histogram(image, roi=(0, 0, 0, 5), bins=16)
    with pytest.raises(ValueError, match="bins"):
        build_histogram(image, roi=(0, 0, 5, 5), bins=100)


# --- backproject ----------------------------------------------------------


def test_backproject_peak_is_one_at_target():
    image = make_test_image()
    prob = backproject(image, _histogram(image))
    assert prob.max() == pytest.approx(1.0)
    assert prob[30, 60] == pytest.approx(1.0)  # red disc centre


def test_backproject_distractor_colour_is_low():
    image = make_test_image()
    prob = backproject(image, _histogram(image))
    assert prob[55, 25] < 0.5  # blue distractor disc centre
    assert prob[0, 0] < 0.5  # green background


def test_backproject_all_zero_histogram_raises():
    image = make_test_image()
    with pytest.raises(ValueError, match="all zeros"):
        backproject(image, np.zeros((8, 8, 8)))


def test_backproject_wrong_shape_histogram_raises():
    image = make_test_image()
    with pytest.raises(ValueError):
        backproject(image, np.ones((8, 8)))
    with pytest.raises(ValueError):
        backproject(image, np.ones((8, 8, 8, 8)))


# --- locate_peak / locate_centroid ----------------------------------------


def test_locate_peak_finds_hot_pixel():
    prob = np.zeros((10, 10))
    prob[3, 7] = 0.9
    assert locate_peak(prob) == (7, 3, 0.9)


def test_locate_centroid_weighted_by_probability():
    prob = np.zeros((10, 10))
    prob[4, 4] = 1.0
    prob[4, 6] = 1.0
    cx, cy = locate_centroid(prob, threshold=0.5)
    assert cx == pytest.approx(5.0)
    assert cy == pytest.approx(4.0)


def test_locate_centroid_ignores_below_threshold():
    prob = np.zeros((10, 10))
    prob[2, 2] = 1.0
    prob[8, 8] = 0.4
    cx, cy = locate_centroid(prob, threshold=0.5)
    assert (cx, cy) == pytest.approx((2.0, 2.0))


def test_locate_centroid_threshold_out_of_range_raises():
    with pytest.raises(ValueError, match="threshold"):
        locate_centroid(np.ones((5, 5)), threshold=1.5)


def test_locate_peak_on_1d_array_raises():
    with pytest.raises(ValueError):
        locate_peak(np.array([1.0, 2.0]))


# --- localise -------------------------------------------------------------


def test_localise_finds_red_disc_centre():
    image = make_test_image()
    result = localise(image, _histogram(image), threshold=0.5)
    cx, cy = result["centroid"]
    assert np.hypot(cx - 60, cy - 30) <= 2.0
    assert result["peak_value"] == pytest.approx(1.0)
    assert result["probability_map"].shape == (80, 100)


def test_localise_peak_inside_disc():
    image = make_test_image()
    result = localise(image, _histogram(image))
    px, py = result["peak"]
    assert (py - 30) ** 2 + (px - 60) ** 2 <= 12 ** 2


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        build_histogram(np.zeros((10, 10)), roi=(0, 0, 5, 5))
    with pytest.raises(ValueError):
        backproject(np.full((5, 5, 3), 300.0), np.ones((4, 4, 4)))
