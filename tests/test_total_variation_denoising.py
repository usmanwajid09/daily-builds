"""Tests for total_variation_denoising (Chambolle)."""
import numpy as np
import pytest

from total_variation_denoising import psnr, tv_denoise
from total_variation_denoising.demo import build_image


def noisy_image(seed=0, sigma=15.0, shape=(32, 32)):
    clean = build_image(*shape) if shape == (48, 64) else _blocks(shape)
    rng = np.random.default_rng(seed)
    return clean, np.clip(clean + rng.normal(0, sigma, size=shape), 0, 255)


def _blocks(shape):
    img = np.full(shape, 90.0)
    img[4:16, 4:16] = 200.0
    img[20:28, 20:28] = 30.0
    return img


def test_denoise_improves_psnr():
    clean, noisy = noisy_image()
    denoised = tv_denoise(noisy, weight=15.0, max_iter=200)
    assert psnr(clean, denoised) > psnr(clean, noisy) + 2


def test_denoise_output_shape_and_range():
    _, noisy = noisy_image()
    out = tv_denoise(noisy)
    assert out.shape == noisy.shape
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_denoise_reduces_noise_on_flat_image():
    rng = np.random.default_rng(2)
    flat = np.clip(128 + rng.normal(0, 15, size=(24, 24)), 0, 255)
    out = tv_denoise(flat, weight=10.0, max_iter=200)
    assert out.std() < flat.std()


def test_edge_preserved():
    clean, noisy = noisy_image()
    out = tv_denoise(noisy, weight=15.0, max_iter=200)
    # The bright block (200) vs background (90) step must survive.
    assert out[8:12, 8:12].mean() - out[8:12, 20:24].mean() > 80


def test_larger_weight_smooths_more():
    _, noisy = noisy_image()
    weak = tv_denoise(noisy, weight=5.0, max_iter=200)
    strong = tv_denoise(noisy, weight=60.0, max_iter=200)
    assert strong.std() < weak.std()


def test_more_iterations_monotone_or_stable():
    _, noisy = noisy_image()
    few = tv_denoise(noisy, weight=15.0, max_iter=5)
    many = tv_denoise(noisy, weight=15.0, max_iter=200)
    # More iterations should not move away from the clean image.
    clean, _ = noisy_image()
    assert psnr(clean, many) >= psnr(clean, few) - 0.5


def test_flat_input_unchanged():
    flat = np.full((16, 16), 100.0)
    out = tv_denoise(flat)
    assert np.allclose(out, flat, atol=1e-6)


def test_uint8_input_accepted():
    clean, noisy = noisy_image()
    out = tv_denoise(noisy.astype(np.uint8))
    assert out.shape == noisy.shape


def test_psnr_identical_is_inf():
    img = build_image()
    assert psnr(img, img) == float("inf")


def test_psnr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4)), np.zeros((4, 5)))


def test_invalid_weight_raises():
    _, noisy = noisy_image()
    with pytest.raises(ValueError):
        tv_denoise(noisy, weight=0)
    with pytest.raises(ValueError):
        tv_denoise(noisy, weight=-5)


def test_invalid_max_iter_and_tol_raise():
    _, noisy = noisy_image()
    with pytest.raises(ValueError):
        tv_denoise(noisy, max_iter=0)
    with pytest.raises(ValueError):
        tv_denoise(noisy, tol=0)


def test_non_array_and_empty_raise():
    with pytest.raises(TypeError):
        tv_denoise([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        tv_denoise(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        tv_denoise(np.zeros((8, 8, 3), dtype=np.uint8))


def test_denoise_deterministic():
    _, noisy = noisy_image()
    assert np.array_equal(tv_denoise(noisy), tv_denoise(noisy))
