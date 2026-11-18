"""Tests for non_local_means."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_local_means import non_local_means, psnr
from non_local_means.demo import make_step_image


def test_constant_image_unchanged():
    image = np.full((10, 12), 77.0)
    out = non_local_means(image)
    assert out.shape == image.shape
    assert np.allclose(out, 77.0)


def test_output_shape_dtype_and_finite():
    rng = np.random.default_rng(0)
    image = rng.random((12, 14)) * 255.0
    out = non_local_means(image)
    assert out.shape == image.shape
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_denoising_reduces_noise():
    clean, noisy = make_step_image()
    out = non_local_means(noisy, search_radius=5, patch_radius=1, h=15.0)
    assert psnr(clean, out) > psnr(clean, noisy) + 2.0
    assert np.std(out[:, :20]) < np.std(noisy[:, :20]) / 2


def test_step_edge_preserved():
    clean, noisy = make_step_image()
    out = non_local_means(noisy, search_radius=5, patch_radius=1, h=15.0)
    step_before = np.mean(noisy[:, 40:]) - np.mean(noisy[:, :20])
    step_after = np.mean(out[:, 40:]) - np.mean(out[:, :20])
    assert abs(step_after - 150.0) < 5.0
    assert abs(step_after - step_before) < 5.0


def test_rgb_image():
    rng = np.random.default_rng(1)
    clean = np.zeros((12, 14, 3))
    clean[:, :7] = [50.0, 100.0, 150.0]
    clean[:, 7:] = [200.0, 80.0, 30.0]
    noisy = clean + rng.normal(0, 12.0, size=clean.shape)
    out = non_local_means(noisy, search_radius=3, patch_radius=1, h=12.0)
    assert out.shape == clean.shape
    assert psnr(clean, out) > psnr(clean, noisy)


def test_large_h_smooths_more_than_small_h():
    clean, noisy = make_step_image()
    out_small = non_local_means(noisy, search_radius=3, patch_radius=1, h=5.0)
    out_large = non_local_means(noisy, search_radius=3, patch_radius=1, h=40.0)
    assert np.std(out_large[:, :20]) < np.std(out_small[:, :20])


def test_patch_radius_zero_is_single_pixel_comparison():
    image = np.full((10, 10), 100.0)
    image[5, 5] = 200.0
    out = non_local_means(image, search_radius=2, patch_radius=0, h=10.0)
    assert out.shape == image.shape
    assert np.isfinite(out).all()


def test_psnr_known_values():
    ref = np.full((4, 4), 100.0)
    assert psnr(ref, ref) == float("inf")
    assert psnr(ref, ref + 10.0) == pytest.approx(20 * np.log10(255.0 / 10.0))
    with pytest.raises(ValueError, match="shapes differ"):
        psnr(ref, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-empty"):
        psnr(np.zeros((0, 0)), np.zeros((0, 0)))
    with pytest.raises(ValueError, match="peak"):
        psnr(ref, ref + 1.0, peak=0)


def test_validation_errors():
    with pytest.raises(ValueError, match="2D.*3D|2D \\(grayscale\\) or 3D"):
        non_local_means(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="channels"):
        non_local_means(np.zeros((8, 8, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        non_local_means(np.zeros((0, 5)))
    with pytest.raises(ValueError, match="finite"):
        non_local_means(np.array([[1.0, np.nan]] * 8))
    with pytest.raises(ValueError, match="search_radius"):
        non_local_means(np.zeros((8, 8)), search_radius=0)
    with pytest.raises(ValueError, match="patch_radius"):
        non_local_means(np.zeros((8, 8)), patch_radius=-1)
    with pytest.raises(ValueError, match="h must be"):
        non_local_means(np.zeros((8, 8)), h=0)
    with pytest.raises(ValueError, match="too small"):
        non_local_means(np.zeros((4, 4)), patch_radius=3)
