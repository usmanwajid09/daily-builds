"""Tests for wiener_deconvolution."""

import numpy as np
import pytest

from wiener_deconvolution import (
    apply_blur,
    estimate_nsr,
    make_gaussian_psf,
    make_motion_psf,
    psnr,
    wiener_deconvolve,
)
from wiener_deconvolution.demo import main as demo_main, make_sharp_image


def test_gaussian_psf_is_normalised_and_centred():
    psf = make_gaussian_psf(size=15, sigma=2.0)
    assert psf.shape == (15, 15)
    assert psf.sum() == pytest.approx(1.0)
    assert np.all(psf >= 0)
    assert psf[7, 7] == psf.max()
    assert np.allclose(psf, psf[::-1, ::-1])  # symmetric


def test_gaussian_psf_validation():
    with pytest.raises(ValueError, match="odd"):
        make_gaussian_psf(size=14)
    with pytest.raises(ValueError, match="positive"):
        make_gaussian_psf(size=15, sigma=0.0)
    with pytest.raises(ValueError, match="positive"):
        make_gaussian_psf(size=0)


def test_motion_psf_is_normalised():
    for angle in (0.0, 45.0, 90.0, 135.0):
        psf = make_motion_psf(length=15, angle_degrees=angle)
        assert psf.shape == (15, 15)
        assert psf.sum() == pytest.approx(1.0)
        assert np.all(psf >= 0)
        # Diagonal lines rasterise with some collisions, so fewer
        # than `length` pixels may be nonzero; each sample still
        # contributes 1/length of the total weight.
        assert 7 < np.count_nonzero(psf) <= 15
        assert np.allclose(psf[np.nonzero(psf)] * 15,
                           np.round(psf[np.nonzero(psf)] * 15))
    with pytest.raises(ValueError, match="odd"):
        make_motion_psf(length=10)


def test_delta_psf_leaves_image_unchanged():
    sharp = make_sharp_image()
    delta = np.zeros((3, 3))
    delta[1, 1] = 1.0
    assert np.allclose(apply_blur(sharp, delta), sharp, atol=1e-9)
    restored = wiener_deconvolve(sharp, delta, nsr=0.0)
    assert np.allclose(restored, sharp, atol=1e-9)


def test_blur_then_deblur_recovers_noise_free_image():
    sharp = make_sharp_image()
    psf = make_gaussian_psf(size=15, sigma=2.0)
    blurred = apply_blur(sharp, psf)
    assert psnr(sharp, blurred) < 30.0  # blur genuinely degrades it
    restored = wiener_deconvolve(blurred, psf, nsr=0.0)
    assert psnr(sharp, restored) > 60.0


def test_wiener_improves_noisy_blurred_image():
    sharp = make_sharp_image()
    psf = make_gaussian_psf(size=15, sigma=2.0)
    blurred = apply_blur(sharp, psf)
    rng = np.random.default_rng(0)
    noisy = blurred + rng.normal(0.0, 1.5, size=sharp.shape)
    nsr = estimate_nsr(1.5 ** 2, float(np.var(sharp)))
    restored = wiener_deconvolve(noisy, psf, nsr=nsr)
    assert restored.shape == sharp.shape
    assert psnr(sharp, restored) > psnr(sharp, noisy) + 2.0


def test_motion_blur_then_deblur():
    sharp = make_sharp_image()
    psf = make_motion_psf(length=9, angle_degrees=0.0)
    blurred = apply_blur(sharp, psf)
    restored = wiener_deconvolve(blurred, psf, nsr=1e-4)
    assert psnr(sharp, restored) > psnr(sharp, blurred)


def test_larger_nsr_means_less_sharpening():
    sharp = make_sharp_image()
    psf = make_gaussian_psf(size=15, sigma=2.0)
    blurred = apply_blur(sharp, psf)
    rng = np.random.default_rng(1)
    noisy = blurred + rng.normal(0.0, 5.0, size=sharp.shape)
    mild = wiener_deconvolve(noisy, psf, nsr=0.001)
    strong = wiener_deconvolve(noisy, psf, nsr=0.1)
    # Stronger regularisation stays closer to the blurred input.
    assert np.mean((strong - noisy) ** 2) < np.mean((mild - noisy) ** 2)


def test_estimate_nsr():
    assert estimate_nsr(4.0, 100.0) == pytest.approx(0.04)
    assert estimate_nsr(0.0, 100.0) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        estimate_nsr(-1.0, 100.0)
    with pytest.raises(ValueError, match="positive"):
        estimate_nsr(1.0, 0.0)


def test_deconvolve_rejects_bad_inputs():
    sharp = make_sharp_image()
    psf = make_gaussian_psf(size=15, sigma=2.0)
    with pytest.raises(ValueError, match="2D"):
        wiener_deconvolve(np.zeros((8, 8, 3)), psf)
    with pytest.raises(ValueError, match="empty"):
        wiener_deconvolve(np.zeros((0, 0)), psf)
    with pytest.raises(ValueError, match="finite"):
        wiener_deconvolve(np.full((8, 8), np.nan), psf)
    with pytest.raises(ValueError, match="non-negative"):
        wiener_deconvolve(sharp, psf, nsr=-0.1)
    with pytest.raises(ValueError, match="odd"):
        wiener_deconvolve(sharp, np.ones((4, 4)))
    with pytest.raises(ValueError, match="sum to zero"):
        wiener_deconvolve(sharp, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="larger than image"):
        wiener_deconvolve(np.zeros((8, 8)), make_gaussian_psf(size=15, sigma=2.0))


def test_blur_rejects_bad_inputs():
    sharp = make_sharp_image()
    with pytest.raises(ValueError, match="larger than image"):
        apply_blur(np.zeros((8, 8)), make_gaussian_psf(size=15, sigma=2.0))
    with pytest.raises(ValueError, match="2D"):
        apply_blur(np.zeros((8, 8, 3)), make_gaussian_psf())


def test_psnr():
    sharp = make_sharp_image()
    assert psnr(sharp, sharp) == float("inf")
    blurred = apply_blur(sharp, make_gaussian_psf(size=15, sigma=2.0))
    value = psnr(sharp, blurred)
    assert 15.0 < value < 35.0
    with pytest.raises(ValueError, match="same shape"):
        psnr(sharp, np.zeros((32, 32)))


def test_demo_runs():
    result = demo_main()
    assert result["psnr_restored"] > result["psnr_blurred"] + 2.0
    assert result["psnr_clean"] > 60.0
