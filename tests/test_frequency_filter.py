"""Tests for frequency_filter (FFT low-pass / high-pass filtering)."""

import numpy as np
import pytest

from frequency_filter.filter import band_pass, high_pass, low_pass, make_mask


def checkerboard(size=64, amp=1.0):
    y, x = np.mgrid[0:size, 0:size]
    return amp * ((x + y) % 2) * 2 - amp


def smooth_gradient(size=64):
    y, x = np.mgrid[0:size, 0:size]
    return (x + y) / (2 * size)


def test_output_shape_matches_input():
    img = np.random.default_rng(0).normal(size=(48, 64))
    assert low_pass(img, 5.0).shape == img.shape
    assert high_pass(img, 5.0).shape == img.shape
    rgb = np.random.default_rng(1).normal(size=(32, 32, 3))
    assert low_pass(rgb, 5.0).shape == rgb.shape
    assert high_pass(rgb, 5.0).shape == rgb.shape


def test_low_pass_reduces_high_frequency_energy():
    img = checkerboard()
    filtered = low_pass(img, 4.0, kind="gaussian")
    assert filtered.std() < 0.5 * img.std()


def test_high_pass_removes_smooth_component():
    # A 1-cycle-per-image sinusoid is purely low frequency: the
    # high-pass (cutoff 4) should remove almost all of it.
    size = 64
    y, x = np.mgrid[0:size, 0:size]
    img = 0.5 * np.sin(2 * np.pi * x / size) + 0.5 * np.cos(2 * np.pi * y / size)
    filtered = high_pass(img, 4.0, kind="gaussian")
    assert abs(filtered.mean()) < 1e-8
    assert filtered.std() < 0.1 * img.std()


def test_low_pass_of_constant_image_returns_constant():
    img = np.full((32, 32), 3.5)
    filtered = low_pass(img, 5.0, kind="gaussian")
    assert np.allclose(filtered, 3.5, atol=1e-8)


def test_ideal_mask_is_binary_and_gaussian_mask_in_range():
    ideal = make_mask((64, 64), 8.0, kind="ideal")
    assert set(np.unique(ideal)).issubset({0.0, 1.0})
    assert ideal[32, 32] == 1.0  # centre (DC) kept
    assert ideal[0, 0] == 0.0  # far corner dropped
    gauss = make_mask((64, 64), 8.0, kind="gaussian")
    assert gauss.min() >= 0.0 and gauss.max() <= 1.0
    assert gauss[32, 32] == pytest.approx(1.0)
    assert gauss[0, 0] < 0.01


def test_invalid_cutoff_raises():
    img = np.ones((16, 16))
    for bad in (0, -1.0, -10.0):
        with pytest.raises(ValueError):
            low_pass(img, bad)
        with pytest.raises(ValueError):
            high_pass(img, bad)
    with pytest.raises(ValueError):
        make_mask((16, 16), 0.0)


def test_invalid_kind_raises():
    img = np.ones((16, 16))
    with pytest.raises(ValueError):
        low_pass(img, 5.0, kind="butterworth")
    with pytest.raises(ValueError):
        make_mask((16, 16), 5.0, kind="wrong")


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        low_pass(np.ones(16), 5.0)  # 1D
    with pytest.raises(ValueError):
        low_pass(np.empty((0, 0)), 5.0)  # empty
    with pytest.raises(ValueError):
        low_pass(np.empty((0, 5)), 5.0)  # empty rows


def test_larger_cutoff_keeps_more_energy():
    img = checkerboard() + smooth_gradient()
    energy = lambda a: float(np.sum(a ** 2))
    small = low_pass(img, 2.0, kind="gaussian")
    large = low_pass(img, 16.0, kind="gaussian")
    assert energy(small) < energy(large) < energy(img)


def test_band_pass_keeps_middle_frequencies():
    img = checkerboard() + smooth_gradient()
    bp = band_pass(img, 2.0, 16.0, kind="gaussian")
    assert bp.shape == img.shape
    assert bp.std() < img.std()  # both extremes removed
    with pytest.raises(ValueError):
        band_pass(img, 16.0, 2.0)  # high must exceed low
