import numpy as np
import pytest

from image_moments.moments import (
    area,
    central_moments,
    centroid,
    hu_moments,
    moments_summary,
    normalised_moments,
    orientation,
    raw_moments,
)
from image_moments.demo import make_test_image


def _rectangle(rows=slice(20, 40), cols=slice(30, 50), shape=(60, 80)):
    image = np.zeros(shape, dtype=np.float64)
    image[rows, cols] = 1.0
    return image


def _single_pixel(shape=(10, 10), y=3, x=7):
    image = np.zeros(shape, dtype=np.float64)
    image[y, x] = 1.0
    return image


# --- raw moments ----------------------------------------------------------


def test_single_pixel_raw_moments():
    m = raw_moments(_single_pixel(), order=3)
    assert m["m00"] == 1.0
    assert m["m10"] == 7.0
    assert m["m01"] == 3.0
    assert m["m20"] == 49.0
    assert m["m02"] == 9.0
    assert m["m11"] == 21.0


def test_rectangle_area_and_m00():
    assert raw_moments(_rectangle(), order=0)["m00"] == 400.0
    assert area(_rectangle()) == 400.0


def test_order_zero_returns_only_m00():
    assert raw_moments(_rectangle(), order=0) == {"m00": 400.0}


def test_grayscale_mass_is_weighted():
    image = np.zeros((5, 5))
    image[2, 2] = 2.5
    assert raw_moments(image, order=0)["m00"] == 2.5


# --- centroid -------------------------------------------------------------


def test_rectangle_centroid():
    cx, cy = centroid(_rectangle())
    assert cx == pytest.approx(39.5)
    assert cy == pytest.approx(29.5)


def test_single_pixel_centroid_is_the_pixel():
    assert centroid(_single_pixel()) == (7.0, 3.0)


def test_all_zero_image_has_no_centroid():
    with pytest.raises(ValueError, match="all-zero"):
        centroid(np.zeros((5, 5)))


# --- central / normalised moments -----------------------------------------


def test_central_moments_about_centroid_are_zero_first_order():
    mu = central_moments(_rectangle(), order=1)
    assert mu["mu00"] == 400.0
    assert mu["mu10"] == pytest.approx(0.0, abs=1e-9)
    assert mu["mu01"] == pytest.approx(0.0, abs=1e-9)


def test_central_moment_of_single_pixel_is_zero():
    mu = central_moments(_single_pixel(), order=2)
    assert mu["mu20"] == 0.0
    assert mu["mu02"] == 0.0
    assert mu["mu11"] == 0.0


def test_normalised_eta00_is_one():
    eta = normalised_moments(_rectangle(), order=3)
    assert eta["eta00"] == pytest.approx(1.0)


def test_normalised_moments_are_scale_invariant():
    small = _rectangle()
    big = np.kron(small, np.ones((2, 2)))
    eta_small = normalised_moments(small, order=3)
    eta_big = normalised_moments(big, order=3)
    for key in eta_small:
        if key == "eta00":
            continue
        assert eta_big[key] == pytest.approx(eta_small[key], rel=0.05)


# --- Hu moments -----------------------------------------------------------


def test_hu_moments_length_seven():
    assert hu_moments(_rectangle()).shape == (7,)


def test_hu_moments_translation_invariant():
    image = make_test_image()
    shifted = np.roll(np.roll(image, 5, axis=0), -8, axis=1)
    assert hu_moments(shifted) == pytest.approx(hu_moments(image), rel=1e-6)


def test_hu_moments_rotation_invariant():
    image = make_test_image()
    rotated = np.rot90(image)
    h1, h2 = hu_moments(image), hu_moments(rotated)
    assert h2 == pytest.approx(h1, rel=0.1)


def test_hu_moments_differ_between_shapes():
    disc = np.zeros((40, 40))
    rows, cols = np.mgrid[0:40, 0:40]
    disc[(rows - 20) ** 2 + (cols - 20) ** 2 <= 10 ** 2] = 1.0
    assert not np.allclose(hu_moments(disc), hu_moments(_rectangle()))


# --- orientation ----------------------------------------------------------


def test_horizontal_bar_orientation_near_zero():
    image = np.zeros((30, 60))
    image[12:18, 5:55] = 1.0
    assert abs(orientation(image)) < 0.05


def test_vertical_bar_orientation_near_pi_over_two():
    image = np.zeros((60, 30))
    image[5:55, 12:18] = 1.0
    assert abs(abs(orientation(image)) - np.pi / 2) < 0.05


# --- summary / validation -------------------------------------------------


def test_moments_summary_keys_and_values():
    summary = moments_summary(_rectangle())
    assert summary["area"] == 400.0
    assert summary["centroid"] == pytest.approx((39.5, 29.5))
    assert len(summary["hu_moments"]) == 7
    assert summary["raw_moments"]["m00"] == 400.0


def test_invalid_order_raises():
    with pytest.raises(ValueError, match="order"):
        raw_moments(_rectangle(), order=4)
    with pytest.raises(ValueError, match="order"):
        raw_moments(_rectangle(), order=-1)
    with pytest.raises(ValueError, match="order"):
        raw_moments(_rectangle(), order=2.5)


def test_invalid_images_raise():
    with pytest.raises(ValueError):
        raw_moments(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError):
        raw_moments(np.zeros((0, 5)))
    with pytest.raises(ValueError):
        raw_moments(np.array([[1.0, -1.0]]))
    with pytest.raises(ValueError):
        raw_moments(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        raw_moments(None)
