"""Tests for color_space_converter (RGB/HSV/YCbCr/grayscale conversions)."""

import numpy as np
import pytest

from color_space_converter.color_space import (
    grayscale_to_rgb,
    hsv_to_rgb,
    rgb_to_grayscale,
    rgb_to_hsv,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)


def _pixel(r, g, b):
    return np.array([[[r, g, b]]], dtype=np.float64)


def test_pure_red_hsv():
    hsv = rgb_to_hsv(_pixel(255, 0, 0))[0, 0]
    np.testing.assert_allclose(hsv, [0.0, 1.0, 1.0], atol=1e-9)


def test_pure_green_hsv():
    hsv = rgb_to_hsv(_pixel(0, 255, 0))[0, 0]
    np.testing.assert_allclose(hsv, [120.0, 1.0, 1.0], atol=1e-9)


def test_pure_blue_hsv():
    hsv = rgb_to_hsv(_pixel(0, 0, 255))[0, 0]
    np.testing.assert_allclose(hsv, [240.0, 1.0, 1.0], atol=1e-9)


def test_yellow_and_cyan_and_magenta_hues():
    assert rgb_to_hsv(_pixel(255, 255, 0))[0, 0, 0] == pytest.approx(60.0)
    assert rgb_to_hsv(_pixel(0, 255, 255))[0, 0, 0] == pytest.approx(180.0)
    assert rgb_to_hsv(_pixel(255, 0, 255))[0, 0, 0] == pytest.approx(300.0)


def test_white_and_black_and_grey_hsv():
    assert rgb_to_hsv(_pixel(255, 255, 255))[0, 0].tolist() == [0.0, 0.0, 1.0]
    assert rgb_to_hsv(_pixel(0, 0, 0))[0, 0].tolist() == [0.0, 0.0, 0.0]
    grey = rgb_to_hsv(_pixel(128, 128, 128))[0, 0]
    assert grey[1] == pytest.approx(0.0)
    assert grey[2] == pytest.approx(128.0 / 255.0)


def test_hsv_to_rgb_primary_colours():
    np.testing.assert_allclose(hsv_to_rgb(np.array([[[0.0, 1.0, 1.0]]]))[0, 0],
                               [255, 0, 0], atol=1e-9)
    np.testing.assert_allclose(hsv_to_rgb(np.array([[[120.0, 1.0, 1.0]]]))[0, 0],
                               [0, 255, 0], atol=1e-9)
    np.testing.assert_allclose(hsv_to_rgb(np.array([[[240.0, 1.0, 1.0]]]))[0, 0],
                               [0, 0, 255], atol=1e-9)


def test_hsv_round_trip_random_colours():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(8, 9, 3)).astype(np.float64)
    back = hsv_to_rgb(rgb_to_hsv(image))
    np.testing.assert_allclose(back, image, atol=1e-9)


def test_white_ycbcr_is_255_128_128():
    np.testing.assert_allclose(rgb_to_ycbcr(_pixel(255, 255, 255))[0, 0],
                               [255, 128, 128], atol=1e-9)


def test_black_ycbcr_is_0_128_128():
    np.testing.assert_allclose(rgb_to_ycbcr(_pixel(0, 0, 0))[0, 0],
                               [0, 128, 128], atol=1e-9)


def test_red_ycbcr_known_values():
    ycbcr = rgb_to_ycbcr(_pixel(255, 0, 0))[0, 0]
    np.testing.assert_allclose(ycbcr, [76.245, 84.972, 255.5], atol=1e-3)


def test_ycbcr_round_trip_random_colours():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(8, 9, 3)).astype(np.float64)
    back = ycbcr_to_rgb(rgb_to_ycbcr(image))
    # The standard inverse uses rounded coefficients (1.402, 0.344136,
    # 0.714136, 1.772), so the round trip is exact only to ~1e-4.
    np.testing.assert_allclose(back, image, atol=1e-3)


def test_grayscale_known_luma_values():
    assert rgb_to_grayscale(_pixel(255, 0, 0))[0, 0] == pytest.approx(76.245)
    assert rgb_to_grayscale(_pixel(0, 255, 0))[0, 0] == pytest.approx(149.685)
    assert rgb_to_grayscale(_pixel(0, 0, 255))[0, 0] == pytest.approx(29.07)
    assert rgb_to_grayscale(_pixel(255, 255, 255))[0, 0] == pytest.approx(255.0)


def test_grayscale_to_rgb_replicates_channels():
    gray = np.array([[0.0, 128.0], [255.0, 64.0]])
    rgb = grayscale_to_rgb(gray)
    assert rgb.shape == (2, 2, 3)
    np.testing.assert_allclose(rgb[..., 0], gray)
    np.testing.assert_allclose(rgb[..., 1], gray)
    np.testing.assert_allclose(rgb[..., 2], gray)


def test_grayscale_round_trip_via_rgb():
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, size=(6, 7)).astype(np.float64)
    np.testing.assert_allclose(rgb_to_grayscale(grayscale_to_rgb(gray)), gray,
                               atol=1e-9)


def test_uint8_input_accepted():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    hsv = rgb_to_hsv(image)
    assert hsv[0, 0, 0] == pytest.approx(0.0)
    assert hsv[0, 0, 1] == pytest.approx(1.0)


def test_output_shapes_preserved():
    image = np.zeros((5, 6, 3))
    assert rgb_to_hsv(image).shape == (5, 6, 3)
    assert rgb_to_ycbcr(image).shape == (5, 6, 3)
    assert rgb_to_grayscale(image).shape == (5, 6)


def test_rejects_2d_rgb_input():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        rgb_to_hsv(np.zeros((4, 4)))


def test_rejects_rgb_value_above_255():
    image = np.zeros((2, 2, 3))
    image[0, 0, 0] = 300.0
    with pytest.raises(ValueError, match="\\[0, 255\\]"):
        rgb_to_hsv(image)


def test_rejects_negative_rgb_value():
    image = np.zeros((2, 2, 3))
    image[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="\\[0, 255\\]"):
        rgb_to_ycbcr(image)


def test_rejects_hue_at_360():
    with pytest.raises(ValueError, match="hue"):
        hsv_to_rgb(np.array([[[360.0, 0.5, 0.5]]]))


def test_rejects_saturation_above_1():
    with pytest.raises(ValueError, match="saturation"):
        hsv_to_rgb(np.array([[[100.0, 1.5, 0.5]]]))


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        rgb_to_hsv(np.zeros((0, 0, 3)))
