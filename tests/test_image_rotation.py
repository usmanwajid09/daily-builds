import numpy as np
import pytest

from image_rotation.rotate import rotate_image


def _single_pixel_image(size=5, row=0, col=4, value=255.0):
    """A black image with one bright pixel."""
    img = np.zeros((size, size))
    img[row, col] = value
    return img


def test_zero_degrees_returns_input():
    img = np.arange(25, dtype=float).reshape(5, 5)
    for method in ("nearest", "bilinear"):
        out = rotate_image(img, 0, method=method)
        assert out.shape == img.shape
        assert np.abs(out - img).max() < 1e-12


def test_360_degrees_equals_zero():
    img = np.arange(25, dtype=float).reshape(5, 5)
    for method in ("nearest", "bilinear"):
        assert np.array_equal(
            rotate_image(img, 360, method=method),
            rotate_image(img, 0, method=method),
        )


def test_90_degrees_moves_corner_pixel():
    # Bright pixel top-right (row 0, col 4); 90 degrees counter-clockwise
    # moves it to the top-left corner (row 0, col 0).
    img = _single_pixel_image()
    for method in ("nearest", "bilinear"):
        out = rotate_image(img, 90, method=method)
        assert out.shape == img.shape
        assert out[0, 0] == pytest.approx(255.0)
        assert out.sum() == pytest.approx(255.0)


def test_180_degrees_moves_corner_pixel_to_opposite_corner():
    img = _single_pixel_image()
    out = rotate_image(img, 180, method="nearest")
    assert out[4, 0] == pytest.approx(255.0)
    assert out.sum() == pytest.approx(255.0)


def test_negative_angle_equals_positive_complement():
    img = np.arange(25, dtype=float).reshape(5, 5)
    assert np.array_equal(rotate_image(img, -45), rotate_image(img, 315))


def test_bilinear_and_nearest_agree_on_integer_grid():
    # At multiples of 90 degrees every source coordinate is an integer,
    # so interpolation is exact and both methods must agree.
    img = np.arange(25, dtype=float).reshape(5, 5)
    for angle in (90, 180, 270):
        assert np.allclose(
            rotate_image(img, angle, method="nearest"),
            rotate_image(img, angle, method="bilinear"),
        )


def test_uniform_image_stays_uniform():
    img = np.full((10, 10), 128.0)
    for angle in (0, 90, 180, 270):
        for method in ("nearest", "bilinear"):
            assert np.allclose(rotate_image(img, angle, method=method), img)
    # At 45 degrees the corners map outside the source (fill), but the
    # centre still samples only real pixels, so it stays uniform.
    out = rotate_image(img, 45, method="bilinear", fill=0.0)
    assert np.allclose(out[4:6, 4:6], 128.0)


def test_fill_value_used_outside_source():
    img = np.zeros((9, 9))
    out = rotate_image(img, 45, method="nearest", fill=255.0)
    # The corners of a 45-degree rotation map outside the source.
    assert out[0, 0] == 255.0
    assert out[0, 8] == 255.0
    assert out[8, 0] == 255.0
    assert out[8, 8] == 255.0
    # The centre still samples the (black) source image.
    assert out[4, 4] == 0.0


def test_rgb_rotated_per_channel():
    rng = np.random.default_rng(0)
    rgb = rng.random((8, 10, 3)) * 255
    out = rotate_image(rgb, 45, method="bilinear")
    assert out.shape == rgb.shape
    assert out.dtype == np.float64
    for c in range(3):
        assert np.allclose(out[:, :, c], rotate_image(rgb[:, :, c], 45))


def test_output_shape_and_dtype():
    img = np.random.default_rng(1).random((7, 11)) * 255
    out = rotate_image(img, 33)
    assert out.shape == img.shape
    assert out.dtype == np.float64
    # Integer input is accepted and returned as float64.
    assert rotate_image(np.zeros((4, 4), dtype=np.uint8), 10).dtype == np.float64


def test_bilinear_smooths_more_than_nearest():
    # A hard diagonal edge rotated 45 degrees: bilinear blends the edge
    # across neighbouring pixels, nearest keeps it crisp.
    img = np.zeros((21, 21))
    idx = np.arange(21)
    img[idx, idx] = 255.0
    near = rotate_image(img, 45, method="nearest")
    bilin = rotate_image(img, 45, method="bilinear")
    # Bilinear spreads the line's energy: fewer pixels at full brightness
    # and more partially-blended pixels than nearest-neighbour.
    assert (bilin >= 254.0).sum() < (near >= 254.0).sum()
    assert ((bilin > 0) & (bilin < 255)).sum() > ((near > 0) & (near < 255)).sum()


def test_invalid_method_raises():
    img = np.zeros((5, 5))
    for bad in ("cubic", "linear", "", None, 1):
        with pytest.raises(ValueError):
            rotate_image(img, 10, method=bad)


def test_invalid_angle_raises():
    img = np.zeros((5, 5))
    for bad in (float("nan"), float("inf"), "45", None, True):
        with pytest.raises(ValueError):
            rotate_image(img, bad)


def test_invalid_fill_raises():
    img = np.zeros((5, 5))
    for bad in (float("nan"), float("inf"), "0", None):
        with pytest.raises(ValueError):
            rotate_image(img, 10, fill=bad)


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        rotate_image(np.zeros((5,)), 0)  # 1D
    with pytest.raises(ValueError):
        rotate_image(np.zeros((0, 5)), 0)  # empty
    with pytest.raises(ValueError):
        rotate_image(np.full((5, 5), np.nan), 0)  # non-finite
    with pytest.raises(ValueError):
        rotate_image(np.full((5, 5), np.inf), 0)  # non-finite
    with pytest.raises(ValueError):
        rotate_image(np.array([["a", "b"]]), 0)  # non-numeric dtype
    with pytest.raises(ValueError):
        rotate_image(np.zeros((5, 5, 4)), 0)  # unsupported channels
    with pytest.raises(ValueError):
        rotate_image(np.zeros((5, 5, 1)), 0)  # wrong channel count
