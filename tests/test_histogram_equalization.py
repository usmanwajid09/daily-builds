import numpy as np
import pytest

from histogram_equalization.demo import make_test_image
from histogram_equalization.equalize import (
    contrast_stretch,
    equalize_histogram,
)


def test_equalized_image_uses_full_range():
    img = make_test_image()
    assert img.max() - img.min() < 60  # genuinely low contrast to start
    out = equalize_histogram(img)
    assert out.min() == 0
    assert out.max() == 255


def test_output_shape_and_dtype():
    img = np.random.default_rng(0).integers(0, 256, size=(9, 13)).astype(np.uint8)
    out = equalize_histogram(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_equalization_flattens_histogram():
    rng = np.random.default_rng(1)
    # Strongly skewed input: most pixels dark.
    img = (rng.beta(0.5, 3.0, size=(60, 60)) * 255).astype(np.uint8)
    out = equalize_histogram(img)

    def _cdf_deviation(hist):
        """Max distance of the empirical CDF from a uniform CDF."""
        cdf = np.cumsum(hist) / hist.sum()
        uniform = np.arange(1, 257) / 256.0
        return np.max(np.abs(cdf - uniform))

    in_hist = np.bincount(img.ravel(), minlength=256)
    out_hist = np.bincount(out.ravel(), minlength=256)
    # Equalization makes the CDF near-linear by construction.
    assert _cdf_deviation(out_hist) < _cdf_deviation(in_hist)


def test_monotonic_mapping_preserves_order():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = equalize_histogram(img)
    # Equalizing a perfectly uniform ramp is the identity map.
    assert np.array_equal(out, img)


def test_constant_image_returned_unchanged():
    img = np.full((6, 6), 77, dtype=np.uint8)
    assert np.array_equal(equalize_histogram(img), img)
    assert np.array_equal(contrast_stretch(img), img)


def test_contrast_stretch_spans_full_range():
    img = make_test_image()
    out = contrast_stretch(img)
    assert out.min() == 0
    assert out.max() == 255


def test_contrast_stretch_is_linear():
    img = np.linspace(50, 150, 100, dtype=np.float64).reshape(10, 10)
    out = contrast_stretch(img, 0, 100)
    # Endpoints map exactly to 0 and 255; the middle maps to ~130
    # (input is rounded to uint8 first: 100.5 -> 101 -> 130).
    assert out[0, 0] == 0
    assert out[-1, -1] == 255
    assert out[5, 0] == pytest.approx(130, abs=2)


def test_contrast_stretch_ignores_outliers():
    img = np.full((10, 10), 100, dtype=np.uint8)
    img[0, 0] = 0    # one dark outlier
    img[9, 9] = 255  # one bright outlier
    out = contrast_stretch(img, 2, 98)
    # Percentiles ignore the single outliers, so the bulk stays put.
    assert out[5, 5] == 100


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        equalize_histogram(np.zeros((5, 5, 3)))  # RGB not supported here
    with pytest.raises(ValueError):
        equalize_histogram(np.zeros((5,)))  # 1D
    with pytest.raises(ValueError):
        equalize_histogram(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        equalize_histogram(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        equalize_histogram(np.array([["a", "b"]]))  # non-numeric
    with pytest.raises(ValueError):
        contrast_stretch(np.zeros((5, 5)), 98, 2)  # low >= high
    with pytest.raises(ValueError):
        contrast_stretch(np.zeros((5, 5)), -1, 98)  # out of range
    with pytest.raises(ValueError):
        contrast_stretch(np.zeros((5, 5)), 0, 101)  # out of range
