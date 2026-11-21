"""Tests for noise_estimator."""

import numpy as np
import pytest

from noise_estimator import (
    estimate_sigma,
    gaussian_denoise,
    noise_level,
    psnr,
    recommend_denoising,
)
from noise_estimator.demo import main as demo_main, make_clean_image


def _noisy(sigma, seed=0, size=64):
    clean = make_clean_image(size=size)
    rng = np.random.default_rng(seed)
    return clean, np.clip(clean + rng.normal(0.0, sigma, size=clean.shape), 0, 255)


def test_immerkaer_estimates_known_sigma():
    for true_sigma in (5.0, 10.0, 12.0, 20.0):
        _, noisy = _noisy(true_sigma)
        est = estimate_sigma(noisy, method="immerkaer")
        assert abs(est - true_sigma) < 0.25 * true_sigma


def test_mad_estimates_known_sigma():
    for true_sigma in (5.0, 10.0, 20.0):
        _, noisy = _noisy(true_sigma)
        est = estimate_sigma(noisy, method="mad")
        assert abs(est - true_sigma) < 0.30 * true_sigma


def test_clean_image_estimates_near_zero():
    clean = make_clean_image()
    assert estimate_sigma(clean, method="immerkaer") < 2.0
    assert estimate_sigma(clean, method="mad") < 2.0


def test_pure_noise_estimate_is_accurate():
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 8.0, size=(64, 64))
    assert abs(estimate_sigma(noise, method="immerkaer") - 8.0) < 0.8
    assert abs(estimate_sigma(noise, method="mad") - 8.0) < 0.8


def test_estimate_is_monotone_in_noise():
    estimates = []
    for sigma in (2.0, 8.0, 16.0):
        _, noisy = _noisy(sigma)
        estimates.append(estimate_sigma(noisy))
    assert estimates[0] < estimates[1] < estimates[2]


def test_estimate_rejects_bad_inputs():
    clean = make_clean_image()
    with pytest.raises(ValueError, match="2D"):
        estimate_sigma(np.zeros((8, 8, 3)))
    with pytest.raises(ValueError, match="empty"):
        estimate_sigma(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="at least 3x3"):
        estimate_sigma(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        estimate_sigma(np.full((8, 8), np.nan))
    with pytest.raises(ValueError, match="method"):
        estimate_sigma(clean, method="wavelet")


def test_noise_level_bands():
    assert noise_level(0.0) == "very low"
    assert noise_level(1.9) == "very low"
    assert noise_level(3.0) == "low"
    assert noise_level(7.0) == "moderate"
    assert noise_level(15.0) == "high"
    assert noise_level(25.0) == "very high"
    with pytest.raises(ValueError, match="non-negative"):
        noise_level(-1.0)


def test_recommend_denoising_methods():
    assert recommend_denoising(1.0)["method"] == "none"
    assert recommend_denoising(3.0)["method"] == "gaussian_blur"
    assert recommend_denoising(3.0)["parameters"]["sigma"] == 0.8
    assert recommend_denoising(7.0)["method"] == "gaussian_blur"
    assert recommend_denoising(7.0)["parameters"]["sigma"] == 1.2
    assert recommend_denoising(15.0)["method"] == "bilateral_filter"
    assert recommend_denoising(30.0)["method"] == "non_local_means"
    rec = recommend_denoising(7.0)
    assert rec["level"] == "moderate"
    assert set(rec) == {"level", "method", "parameters", "note"}
    with pytest.raises(ValueError, match="non-negative"):
        recommend_denoising(-2.0)


def test_gaussian_denoise_reduces_noise():
    clean, noisy = _noisy(12.0)
    denoised = gaussian_denoise(noisy, sigma=1.2)
    assert denoised.shape == clean.shape
    assert psnr(clean, denoised) > psnr(clean, noisy)


def test_gaussian_denoise_preserves_flat_image():
    flat = np.full((16, 16), 128.0)
    assert np.allclose(gaussian_denoise(flat, sigma=1.0), flat, atol=1e-9)


def test_gaussian_denoise_validation():
    clean = make_clean_image()
    with pytest.raises(ValueError, match="positive"):
        gaussian_denoise(clean, sigma=0.0)
    with pytest.raises(ValueError, match="2D"):
        gaussian_denoise(np.zeros((8, 8, 3)))


def test_psnr():
    clean = make_clean_image()
    assert psnr(clean, clean) == float("inf")
    _, noisy = _noisy(10.0)
    value = psnr(clean, noisy)
    assert 20.0 < value < 35.0
    with pytest.raises(ValueError, match="same shape"):
        psnr(clean, np.zeros((32, 32)))


def test_demo_runs():
    result = demo_main()
    assert abs(result["immerkaer"] - result["true_sigma"]) < 3.0
    assert abs(result["mad"] - result["true_sigma"]) < 3.0
    assert result["psnr_gain"] > 0
