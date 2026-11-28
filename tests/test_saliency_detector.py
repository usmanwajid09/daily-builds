"""Tests for saliency_detector."""

import numpy as np
import pytest

from saliency_detector import saliency_peak, spectral_residual_saliency
from saliency_detector.demo import make_test_image


def test_peak_finds_planted_disk():
    image, centre = make_test_image()
    saliency = spectral_residual_saliency(image)
    peak = saliency_peak(saliency)
    # Spectral residual flags the disk's rim, so the peak lands on the
    # disk (within its radius) rather than at its centre.
    assert np.hypot(peak[0] - centre[0], peak[1] - centre[1]) <= 8.0


def test_output_range_and_shape():
    image, _ = make_test_image()
    saliency = spectral_residual_saliency(image)
    assert saliency.shape == image.shape
    assert saliency.dtype == np.float64
    assert saliency.min() >= 0.0 and saliency.max() <= 1.0
    assert saliency.max() == pytest.approx(1.0)


def test_constant_image_has_no_salient_spot():
    # A featureless image has no salient location: the map is flat.
    image = np.full((16, 16), 128.0)
    saliency = spectral_residual_saliency(image)
    assert saliency.shape == image.shape
    assert np.ptp(saliency) < 1e-3


def test_peak_is_argmax():
    rng = np.random.default_rng(1)
    saliency = rng.random((7, 9))
    assert saliency_peak(saliency) == tuple(
        int(v) for v in np.unravel_index(np.argmax(saliency), saliency.shape))


def test_saliency_higher_at_disk_than_background():
    image, centre = make_test_image()
    saliency = spectral_residual_saliency(image)
    yy, xx = np.mgrid[0:64, 0:64]
    disk = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= 8 ** 2
    assert saliency[disk].mean() > saliency[~disk].mean()


def test_invalid_input_rejected():
    with pytest.raises(ValueError, match="2D"):
        spectral_residual_saliency(np.zeros((5, 5, 3)))
    with pytest.raises(ValueError, match="empty"):
        spectral_residual_saliency(np.zeros((0, 0)))


def test_invalid_peak_input_rejected():
    with pytest.raises(ValueError, match="2D"):
        saliency_peak(np.zeros((5,)))


def test_deterministic():
    image, _ = make_test_image()
    a = spectral_residual_saliency(image)
    b = spectral_residual_saliency(image)
    assert np.array_equal(a, b)
