import numpy as np
import pytest

from phase_correlation.phase_correlation import (
    estimate_translation,
    phase_correlate,
    shift_image_fourier,
)
from phase_correlation.demo import make_test_image


# --- shift_image_fourier --------------------------------------------------


def test_fourier_shift_by_zero_is_identity():
    image = make_test_image()
    assert shift_image_fourier(image, 0.0, 0.0) == pytest.approx(image, abs=1e-9)


def test_fourier_shift_roundtrip_returns_original():
    image = make_test_image()
    shifted = shift_image_fourier(image, 2.0, 3.0)
    back = shift_image_fourier(shifted, -2.0, -3.0)
    assert back == pytest.approx(image, abs=1e-8)


def test_fourier_shift_matches_integer_roll():
    image = make_test_image()
    shifted = shift_image_fourier(image, 4.0, -3.0)
    rolled = np.roll(np.roll(image, 4, axis=0), -3, axis=1)
    assert shifted == pytest.approx(rolled, abs=1e-8)


def test_fourier_shift_nonfinite_raises():
    with pytest.raises(ValueError):
        shift_image_fourier(make_test_image(), np.nan, 0.0)


# --- phase_correlate ------------------------------------------------------


def test_identical_images_give_zero_shift_and_peak_one():
    image = make_test_image()
    dy, dx, peak = phase_correlate(image, image)
    assert dy == pytest.approx(0.0, abs=0.05)
    assert dx == pytest.approx(0.0, abs=0.05)
    assert peak == pytest.approx(1.0, abs=0.02)


def test_integer_roll_shift_recovered():
    image = make_test_image()
    shifted = np.roll(np.roll(image, 5, axis=0), -3, axis=1)
    dy, dx, peak = phase_correlate(image, shifted, apply_window=False)
    assert dy == pytest.approx(5.0, abs=0.2)
    assert dx == pytest.approx(-3.0, abs=0.2)
    assert peak > 0.9


def test_negative_integer_shift_recovered():
    image = make_test_image()
    shifted = np.roll(np.roll(image, -7, axis=0), 6, axis=1)
    dy, dx, _ = phase_correlate(image, shifted, apply_window=False)
    assert dy == pytest.approx(-7.0, abs=0.2)
    assert dx == pytest.approx(6.0, abs=0.2)


def test_subpixel_shift_recovered():
    image = make_test_image()
    shifted = shift_image_fourier(image, 3.5, -2.25)
    dy, dx, peak = phase_correlate(image, shifted, subpixel=True)
    assert dy == pytest.approx(3.5, abs=0.4)
    assert dx == pytest.approx(-2.25, abs=0.4)
    assert peak > 0.5


def test_subpixel_off_gives_near_integer_shift():
    image = make_test_image()
    shifted = shift_image_fourier(image, 2.0, -4.0)
    dy, dx, _ = phase_correlate(image, shifted, subpixel=False)
    assert dy == pytest.approx(2.0, abs=0.05)
    assert dx == pytest.approx(-4.0, abs=0.05)


def test_windowed_correlation_still_recovers_shift():
    image = make_test_image()
    shifted = shift_image_fourier(image, -2.5, 3.0)
    dy, dx, _ = phase_correlate(image, shifted, apply_window=True)
    assert dy == pytest.approx(-2.5, abs=0.4)
    assert dx == pytest.approx(3.0, abs=0.4)


def test_reverse_shift_is_negated():
    image = make_test_image()
    shifted = shift_image_fourier(image, 2.5, 1.5)
    dy1, dx1, _ = phase_correlate(image, shifted)
    dy2, dx2, _ = phase_correlate(shifted, image)
    assert dy2 == pytest.approx(-dy1, abs=0.2)
    assert dx2 == pytest.approx(-dx1, abs=0.2)


# --- estimate_translation / validation ------------------------------------


def test_estimate_translation_dict_contents():
    image = make_test_image()
    shifted = shift_image_fourier(image, 1.0, 2.0)
    result = estimate_translation(image, shifted)
    assert result["dy"] == pytest.approx(1.0, abs=0.3)
    assert result["dx"] == pytest.approx(2.0, abs=0.3)
    assert result["shift"] == (result["dy"], result["dx"])
    assert result["magnitude"] == pytest.approx(np.hypot(1.0, 2.0), abs=0.3)
    assert 0.0 <= result["peak"] <= 1.0


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        phase_correlate(np.zeros((10, 10)), np.zeros((12, 10)))


def test_constant_image_raises():
    image = make_test_image()
    with pytest.raises(ValueError, match="non-constant"):
        phase_correlate(np.full((64, 64), 5.0), image)
    with pytest.raises(ValueError, match="non-constant"):
        phase_correlate(image, np.zeros((64, 64)))


def test_invalid_images_raise():
    image = make_test_image()
    with pytest.raises(ValueError):
        phase_correlate(np.zeros((5, 5, 3)), image)
    with pytest.raises(ValueError):
        phase_correlate(np.array([[1.0, np.inf]]), np.ones((1, 2)))
    with pytest.raises(ValueError):
        phase_correlate(None, image)
