"""Tests for wavelet_denoising (Haar DWT denoising)."""
import numpy as np
import pytest

from wavelet_denoising import haar_dwt2, haar_idwt2, psnr, wavelet_denoise
from wavelet_denoising.demo import build_image


def noisy_image(seed=0, sigma=20.0, shape=(32, 32)):
    clean = build_image(*shape)
    rng = np.random.default_rng(seed)
    return clean, np.clip(clean + rng.normal(0, sigma, size=shape), 0, 255)


def test_dwt_idwt_roundtrip():
    img = build_image(32, 32)
    coeffs, orig = haar_dwt2(img, levels=2)
    restored = haar_idwt2(coeffs, orig, levels=2)
    assert restored.shape == img.shape
    assert np.allclose(restored, img, atol=1e-9)


def test_dwt_idwt_roundtrip_odd_size():
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 255, size=(30, 46))
    coeffs, orig = haar_dwt2(img, levels=2)
    restored = haar_idwt2(coeffs, orig, levels=2)
    assert restored.shape == img.shape
    assert np.allclose(restored, img, atol=1e-9)


def test_dwt_levels_one_and_three():
    img = build_image(32, 32)
    for levels in (1, 3):
        coeffs, orig = haar_dwt2(img, levels=levels)
        assert np.allclose(haar_idwt2(coeffs, orig, levels=levels), img,
                           atol=1e-9)


def test_denoise_improves_psnr_soft():
    clean, noisy = noisy_image()
    denoised = wavelet_denoise(noisy, levels=2, mode="soft")
    assert psnr(clean, denoised) > psnr(clean, noisy) + 3


def test_denoise_improves_psnr_hard():
    clean, noisy = noisy_image()
    denoised = wavelet_denoise(noisy, levels=2, mode="hard")
    assert psnr(clean, denoised) > psnr(clean, noisy)


def test_denoise_output_dtype_and_range():
    _, noisy = noisy_image()
    out = wavelet_denoise(noisy)
    assert out.dtype == np.uint8
    assert out.shape == noisy.shape
    assert out.min() >= 0 and out.max() <= 255


def test_denoise_explicit_threshold():
    clean, noisy = noisy_image()
    out = wavelet_denoise(noisy, threshold=30.0)
    assert psnr(clean, out) > psnr(clean, noisy)


def test_denoise_flat_noisy_image_becomes_flat():
    rng = np.random.default_rng(1)
    flat = np.clip(128 + rng.normal(0, 15, size=(32, 32)), 0, 255)
    out = wavelet_denoise(flat, levels=2)
    assert out.std() < flat.std()
    assert abs(out.mean() - 128) < 5


def test_psnr_identical_is_inf():
    img = build_image(16, 16)
    assert psnr(img, img) == float("inf")


def test_psnr_known_value():
    ref = np.zeros((4, 4))
    other = np.full((4, 4), 10.0)
    # MSE = 100 -> PSNR = 10*log10(255^2/100) ~= 28.13 dB
    assert abs(psnr(ref, other) - 28.13) < 0.05


def test_psnr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4)), np.zeros((4, 5)))


def test_invalid_mode_raises():
    _, noisy = noisy_image()
    with pytest.raises(ValueError):
        wavelet_denoise(noisy, mode="median")
    with pytest.raises(ValueError):
        wavelet_denoise(noisy, threshold=-1)


def test_invalid_levels_raises():
    img = build_image(16, 16)
    with pytest.raises(ValueError):
        haar_dwt2(img, levels=0)
    with pytest.raises(ValueError):
        haar_idwt2(np.zeros((16, 16)), (16, 16), levels=0)


def test_non_array_and_empty_raise():
    with pytest.raises(TypeError):
        wavelet_denoise([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        wavelet_denoise(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        wavelet_denoise(np.zeros((8, 8, 3), dtype=np.uint8))


def test_denoise_deterministic():
    _, noisy = noisy_image()
    a = wavelet_denoise(noisy)
    b = wavelet_denoise(noisy)
    assert np.array_equal(a, b)
