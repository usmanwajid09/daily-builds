"""Tests for image_registration."""

import numpy as np
import pytest

from image_registration import (
    apply_shift,
    estimate_shift,
    phase_correlation,
    register_translation,
)
from image_registration.demo import make_pattern


def test_zero_shift():
    reference = make_pattern()
    shift, peak = phase_correlation(reference, reference)
    assert shift == (0.0, 0.0)
    assert peak == pytest.approx(1.0, abs=0.01)


def test_integer_shift_recovered():
    reference = make_pattern()
    for true_shift in [(5, -7), (-3, 4), (0, 9), (-12, 0), (8, 8)]:
        moving = apply_shift(reference, *true_shift)
        shift, peak = phase_correlation(reference, moving)
        assert shift == (float(true_shift[0]), float(true_shift[1]))
        assert peak > 0.9


def test_negative_shift_recovered():
    reference = make_pattern()
    moving = apply_shift(reference, -6, -11)
    assert estimate_shift(reference, moving) == (-6.0, -11.0)


def test_subpixel_shift_recovered():
    reference = make_pattern()
    true_shift = (2.5, -3.5)
    moving = apply_shift(reference, *true_shift)
    shift, peak = phase_correlation(reference, moving, subpixel=True)
    assert abs(shift[0] - true_shift[0]) < 0.1
    assert abs(shift[1] - true_shift[1]) < 0.1
    assert peak > 0.2  # a fractional shift spreads the peak over pixels


def test_subpixel_quarter_shift():
    reference = make_pattern()
    true_shift = (1.25, 2.75)
    moving = apply_shift(reference, *true_shift)
    shift, _ = phase_correlation(reference, moving, subpixel=True)
    assert abs(shift[0] - true_shift[0]) < 0.15
    assert abs(shift[1] - true_shift[1]) < 0.15


def test_registered_matches_reference():
    reference = make_pattern()
    moving = apply_shift(reference, 5, -7)
    result = register_translation(reference, moving)
    assert result["shift"] == (5.0, -7.0)
    assert result["peak"] > 0.9
    assert np.abs(result["registered"] - reference).max() < 1e-10


def test_register_translation_subpixel():
    reference = make_pattern()
    moving = apply_shift(reference, 2.5, -3.5)
    result = register_translation(reference, moving, subpixel=True)
    assert abs(result["shift"][0] - 2.5) < 0.1
    assert abs(result["shift"][1] - (-3.5)) < 0.1
    # The estimate is subpixel-accurate, so the registered image is
    # nearly the reference (a half-pixel shift of white noise loses
    # its Nyquist component, so allow a small residual).
    corr = np.corrcoef(result["registered"].ravel(),
                       reference.ravel())[0, 1]
    assert corr > 0.97
    assert np.abs(result["registered"] - reference).max() < 1.0


def test_apply_shift_integer_is_roll():
    reference = make_pattern()
    shifted = apply_shift(reference, 3, -4)
    assert np.array_equal(shifted, np.roll(reference, (3, -4), axis=(0, 1)))


def make_smooth_pattern(seed=0):
    """Bandlimited pattern (no Nyquist content) for exact roundtrips."""
    rng = np.random.default_rng(seed)
    return np.kron(rng.normal(0.0, 1.0, size=(8, 8)), np.ones((8, 8)))


def test_apply_shift_fractional_roundtrip():
    # Exact on a bandlimited pattern (nothing lives at Nyquist).
    reference = make_smooth_pattern()
    shifted = apply_shift(reference, 2.5, -3.5)
    back = apply_shift(shifted, -2.5, 3.5)
    assert np.abs(back - reference).max() < 1e-10
    # On white noise a half-pixel shift zeroes the Nyquist component
    # (cos(pi * 2.5) = 0), so the roundtrip keeps ~98% correlation
    # but not the last bit of Nyquist energy.
    noisy = make_pattern()
    shifted = apply_shift(noisy, 2.5, -3.5)
    back = apply_shift(shifted, -2.5, 3.5)
    corr = np.corrcoef(back.ravel(), noisy.ravel())[0, 1]
    assert corr > 0.98


def test_mismatched_shapes_raise():
    reference = make_pattern()
    with pytest.raises(ValueError):
        phase_correlation(reference, np.zeros((32, 32)))


def test_constant_image_raises():
    reference = make_pattern()
    with pytest.raises(ValueError):
        phase_correlation(np.full((64, 64), 5.0), reference)
    with pytest.raises(ValueError):
        phase_correlation(reference, np.full((64, 64), 5.0))


def test_rejects_bad_shapes():
    reference = make_pattern()
    with pytest.raises(ValueError):
        phase_correlation(reference, np.zeros((64, 64, 3)))
    with pytest.raises(ValueError):
        phase_correlation(np.zeros((0, 0)), np.zeros((0, 0)))


def test_unrelated_images_low_peak():
    reference = make_pattern(seed=0)
    other = make_pattern(seed=99)
    _, peak = phase_correlation(reference, other)
    assert peak < 0.6


def test_estimate_shift_wrapper():
    reference = make_pattern()
    moving = apply_shift(reference, -4, 6)
    assert estimate_shift(reference, moving) == (-4.0, 6.0)
