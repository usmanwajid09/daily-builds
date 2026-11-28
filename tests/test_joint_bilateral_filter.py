"""Tests for joint_bilateral_filter."""

import numpy as np
import pytest

from joint_bilateral_filter import gaussian_kernel_2d, joint_bilateral_filter
from joint_bilateral_filter.demo import make_test_images


def test_constant_image_unchanged():
    image = np.full((10, 12), 77.0)
    out = joint_bilateral_filter(image, image)
    assert out.shape == image.shape
    assert np.allclose(out, 77.0)


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    image = rng.random((9, 11)) * 255.0
    guide = rng.random((9, 11)) * 255.0
    out = joint_bilateral_filter(image, guide)
    assert out.shape == image.shape
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_clean_guide_beats_self_guide():
    clean, noisy = make_test_images()
    flat = (slice(None), slice(0, 25))
    guided = joint_bilateral_filter(noisy, clean, diameter=5,
                                    sigma_color=25.0, sigma_space=2.0)
    plain = joint_bilateral_filter(noisy, noisy, diameter=5,
                                   sigma_color=25.0, sigma_space=2.0)
    assert guided[flat].std() < plain[flat].std()
    assert guided[flat].std() < noisy[flat].std() * 0.5


def test_step_preserved_with_clean_guide():
    clean, noisy = make_test_images()
    guided = joint_bilateral_filter(noisy, clean, diameter=5,
                                    sigma_color=25.0, sigma_space=2.0)
    step_before = noisy[:, 35].mean() - noisy[:, 24].mean()
    step_after = guided[:, 35].mean() - guided[:, 24].mean()
    assert abs(step_after - step_before) < 10.0


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="same shape"):
        joint_bilateral_filter(np.zeros((5, 5)), np.zeros((6, 5)))


def test_invalid_diameter_and_sigma_rejected():
    image = np.zeros((5, 5))
    with pytest.raises(ValueError, match="odd"):
        joint_bilateral_filter(image, image, diameter=4)
    with pytest.raises(ValueError, match="positive"):
        joint_bilateral_filter(image, image, sigma_color=0.0)


def test_non_2d_rejected():
    with pytest.raises(ValueError, match="2D"):
        joint_bilateral_filter(np.zeros((5, 5, 3)), np.zeros((5, 5, 3)))


def test_gaussian_kernel_normalised():
    kernel = gaussian_kernel_2d(5, 1.0)
    assert kernel.shape == (5, 5)
    assert abs(kernel.sum() - 1.0) < 1e-12
    assert kernel[2, 2] == kernel.max()
