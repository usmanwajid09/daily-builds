"""Tests for demosaicing (Bayer demosaicing)."""

import numpy as np
import pytest

from demosaicing.demosaicing import demosaic, mosaic, psnr


def _smooth_rgb(height=20, width=24):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    red = 255.0 * np.exp(-((yy - 10.0) ** 2 + (xx - 12.0) ** 2) / 200.0)
    green = 255.0 * (xx / width)
    blue = 255.0 * (yy / height)
    return np.stack([red, green, blue], axis=2)


def test_round_trip_psnr_high_on_smooth_image():
    rgb = _smooth_rgb()
    recovered = demosaic(mosaic(rgb, "RGGB"), "RGGB")
    assert psnr(rgb, recovered) > 30.0


def test_output_shape_and_dtype():
    bayer = mosaic(_smooth_rgb(), "RGGB")
    recovered = demosaic(bayer, "RGGB")
    assert recovered.shape == (20, 24, 3)
    assert recovered.dtype == np.float64


def test_constant_image_reconstructed_exactly():
    rgb = np.full((8, 10, 3), [40.0, 120.0, 200.0])
    recovered = demosaic(mosaic(rgb, "RGGB"), "RGGB")
    np.testing.assert_allclose(recovered, rgb, atol=1e-9)


def test_measured_pixels_keep_exact_values():
    rgb = _smooth_rgb()
    bayer = mosaic(rgb, "RGGB")
    recovered = demosaic(bayer, "RGGB")
    # RGGB tile: R at (0,0), G at (0,1) and (1,0), B at (1,1).
    assert recovered[0, 0, 0] == pytest.approx(rgb[0, 0, 0])
    assert recovered[0, 1, 1] == pytest.approx(rgb[0, 1, 1])
    assert recovered[1, 0, 1] == pytest.approx(rgb[1, 0, 1])
    assert recovered[1, 1, 2] == pytest.approx(rgb[1, 1, 2])


def test_all_four_patterns_round_trip():
    rgb = _smooth_rgb()
    for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
        recovered = demosaic(mosaic(rgb, pattern), pattern)
        assert psnr(rgb, recovered) > 30.0, pattern


def test_mosaic_samples_correct_channels():
    rgb = np.zeros((4, 4, 3))
    rgb[:, :, 0] = 10.0
    rgb[:, :, 1] = 20.0
    rgb[:, :, 2] = 30.0
    bayer = mosaic(rgb, "RGGB")
    assert bayer[0, 0] == 10.0  # R
    assert bayer[0, 1] == 20.0  # G
    assert bayer[1, 0] == 20.0  # G
    assert bayer[1, 1] == 30.0  # B
    bayer_bggr = mosaic(rgb, "BGGR")
    assert bayer_bggr[0, 0] == 30.0  # B
    assert bayer_bggr[1, 1] == 10.0  # R


def test_uint8_bayer_accepted():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 100
    rgb[:, :, 1] = 150
    rgb[:, :, 2] = 200
    recovered = demosaic(mosaic(rgb, "RGGB"), "RGGB")
    np.testing.assert_allclose(recovered, rgb.astype(np.float64), atol=1e-9)


def test_psnr_identical_images_is_inf():
    image = _smooth_rgb()
    assert psnr(image, image) == float("inf")


def test_psnr_known_value():
    ref = np.zeros((2, 2))
    other = np.full((2, 2), 1.0)
    # MSE = 1, max 255 -> 10*log10(255^2) ~= 48.13 dB.
    assert psnr(ref, other) == pytest.approx(48.1308, abs=1e-3)


def test_rejects_odd_bayer_dimensions():
    with pytest.raises(ValueError, match="even"):
        demosaic(np.zeros((5, 6)))


def test_rejects_odd_rgb_dimensions():
    with pytest.raises(ValueError, match="even"):
        mosaic(np.zeros((5, 6, 3)))


def test_rejects_1d_bayer():
    with pytest.raises(ValueError, match="2D"):
        demosaic(np.zeros(10))


def test_rejects_3d_bayer():
    with pytest.raises(ValueError, match="2D"):
        demosaic(np.zeros((4, 4, 3)))


def test_rejects_rgb_with_wrong_channels():
    with pytest.raises(ValueError, match="HxWx3"):
        mosaic(np.zeros((4, 4, 4)))


def test_rejects_empty_bayer():
    with pytest.raises(ValueError, match="empty"):
        demosaic(np.zeros((0, 0)))


def test_rejects_nan_bayer():
    bayer = np.zeros((4, 4))
    bayer[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        demosaic(bayer)


def test_rejects_unknown_pattern():
    with pytest.raises(ValueError, match="unknown Bayer pattern"):
        demosaic(np.zeros((4, 4)), pattern="RGBG")
    with pytest.raises(ValueError, match="unknown Bayer pattern"):
        mosaic(np.zeros((4, 4, 3)), pattern="RGBG")


def test_psnr_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr(np.zeros((2, 2)), np.zeros((3, 3)))
