import numpy as np
import pytest

from color_histogram.demo import make_blue_image, make_red_image
from color_histogram.hist import (
    backproject,
    channel_histograms,
    compare_histograms,
)


def _solid_rgb(value, shape=(4, 5, 3)):
    return np.full(shape, value, dtype=np.float64)


def test_histograms_normalised_and_correct_length():
    hists = channel_histograms(_solid_rgb(128.0), bins=16)
    assert len(hists) == 3
    for h in hists:
        assert h.shape == (16,)
        assert h.sum() == pytest.approx(1.0)


def test_histogram_peaks_at_pixel_value():
    img = np.zeros((4, 4, 3))
    img[:, :, 0] = 10.0
    img[:, :, 1] = 100.0
    img[:, :, 2] = 250.0
    hists = channel_histograms(img, bins=256, value_range=(0, 255))
    assert hists[0].argmax() == 10
    assert hists[1].argmax() == 100
    assert hists[2].argmax() == 250
    assert hists[0][10] == pytest.approx(1.0)


def test_histogram_counts_match_pixel_mix():
    img = np.zeros((2, 2, 3))
    img[0, 0, 0] = 0.0
    img[0, 1, 0] = 0.0
    img[1, 0, 0] = 255.0
    img[1, 1, 0] = 255.0
    h = channel_histograms(img, bins=2)[0]
    assert h == pytest.approx([0.5, 0.5])


def test_compare_with_itself():
    h = channel_histograms(make_red_image())[0]
    assert compare_histograms(h, h, "correlation") == pytest.approx(1.0)
    assert compare_histograms(h, h, "chi_square") == pytest.approx(0.0)
    assert compare_histograms(h, h, "intersection") == pytest.approx(1.0)


def test_compare_different_images():
    red = channel_histograms(make_red_image())[0]
    blue = channel_histograms(make_blue_image())[0]
    assert compare_histograms(red, blue, "correlation") < 0.0
    assert compare_histograms(red, blue, "chi_square") > 0.0
    assert compare_histograms(red, blue, "intersection") == pytest.approx(0.0)


def test_compare_is_symmetric():
    a = np.array([0.5, 0.3, 0.2])
    b = np.array([0.1, 0.4, 0.5])
    for method in ("correlation", "chi_square", "intersection"):
        assert compare_histograms(a, b, method) == pytest.approx(
            compare_histograms(b, a, method)
        )


def test_intersection_known_value():
    a = np.array([0.5, 0.5, 0.0])
    b = np.array([0.25, 0.25, 0.5])
    assert compare_histograms(a, b, "intersection") == pytest.approx(0.5)


def test_chi_square_known_value():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    # (1-0)^2/1 + (0-1)^2/1 = 2
    assert compare_histograms(a, b, "chi_square") == pytest.approx(2.0)


def test_backproject_shape_and_range():
    img = make_red_image()
    hist = channel_histograms(img)[0]
    prob = backproject(img, hist)
    assert prob.shape == img.shape[:2]
    assert prob.dtype == np.float64
    assert prob.min() >= 0.0
    assert prob.max() <= hist.max()


def test_backproject_matches_own_histogram_best():
    red = make_red_image()
    blue = make_blue_image()
    hist = channel_histograms(red)[0]
    assert backproject(red, hist).mean() > backproject(blue, hist).mean()


def test_backproject_single_spike_histogram():
    img = np.zeros((3, 3, 3))
    img[:, :, :] = 50.0
    img[1, 1, :] = 200.0
    hist = np.zeros(256)
    hist[200] = 1.0
    prob = backproject(img, hist)
    assert prob[1, 1] == pytest.approx(1.0)
    assert prob[0, 0] == pytest.approx(0.0)


def test_backproject_channel_parameter():
    img = np.zeros((2, 2, 3))
    img[:, :, 0] = 200.0  # red channel high
    img[:, :, 2] = 50.0   # blue channel low
    hist = np.zeros(256)
    hist[200] = 1.0
    assert backproject(img, hist, channel=0).mean() == pytest.approx(1.0)
    assert backproject(img, hist, channel=2).mean() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        backproject(img, hist, channel=3)
    with pytest.raises(ValueError):
        backproject(img, hist, channel="red")


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        channel_histograms(np.zeros((5, 5)))  # 2D, not RGB
    with pytest.raises(ValueError):
        channel_histograms(np.zeros((5, 5, 4)))  # 4 channels
    with pytest.raises(ValueError):
        channel_histograms(np.zeros((0, 5, 3)))  # empty
    with pytest.raises(ValueError):
        channel_histograms(np.full((2, 2, 3), np.nan))  # non-finite
    with pytest.raises(ValueError):
        channel_histograms(np.array([[["a"] * 3] * 2] * 2))  # non-numeric
    with pytest.raises(ValueError):
        backproject(np.zeros((5, 5)), np.ones(256))  # 2D image


def test_invalid_bins_and_range_raise():
    img = _solid_rgb(10.0)
    for bad in (0, -4, 2.5, "256", True):
        with pytest.raises(ValueError):
            channel_histograms(img, bins=bad)
    for bad_range in ((255, 0), (10, 10), (0, np.inf), (0, np.nan), (5,)):
        with pytest.raises(ValueError):
            channel_histograms(img, value_range=bad_range)


def test_invalid_histogram_raises():
    with pytest.raises(ValueError):
        compare_histograms([0.5, 0.5], [0.5])  # mismatched lengths
    with pytest.raises(ValueError):
        compare_histograms([0.5, -0.5], [0.5, 0.5])  # negative
    with pytest.raises(ValueError):
        compare_histograms([], [])  # empty
    with pytest.raises(ValueError):
        compare_histograms([0.5, np.nan], [0.5, 0.5])  # non-finite
    with pytest.raises(ValueError):
        compare_histograms(np.ones((2, 2)), np.ones((2, 2)))  # not 1D
    with pytest.raises(ValueError):
        backproject(_solid_rgb(10.0), np.ones(128))  # length != bins


def test_unknown_method_raises():
    h = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="unknown method"):
        compare_histograms(h, h, method="euclidean")


def test_bins_parameter_respected():
    img = _solid_rgb(200.0)
    hists = channel_histograms(img, bins=8, value_range=(0, 255))
    assert hists[0].shape == (8,)
    assert hists[0].argmax() == 6  # 200/256*8 = 6.25 -> bin 6
    prob = backproject(img, hists[0], bins=8)
    assert prob.shape == img.shape[:2]
    assert prob.mean() == pytest.approx(1.0)
