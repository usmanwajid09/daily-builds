import numpy as np
import pytest

from gaussian_blur.blur import gaussian_blur, gaussian_kernel_1d
from gaussian_blur.demo import make_test_image


def test_kernel_sums_to_one_and_is_symmetric():
    for sigma in (0.5, 1.0, 2.5):
        kernel = gaussian_kernel_1d(sigma)
        assert kernel.sum() == pytest.approx(1.0)
        assert np.allclose(kernel, kernel[::-1])
        assert kernel[len(kernel) // 2] == kernel.max()


def test_kernel_size_defaults_to_cover_3_sigma():
    assert len(gaussian_kernel_1d(1.0)) == 7   # 2*ceil(3)+1
    assert len(gaussian_kernel_1d(2.0)) == 13  # 2*ceil(6)+1
    assert len(gaussian_kernel_1d(1.0, kernel_size=5)) == 5


def test_constant_image_is_unchanged():
    img = np.full((8, 8), 128.0)
    assert np.allclose(gaussian_blur(img, sigma=1.0), img)
    # Replicate padding keeps the border flat too.
    assert np.allclose(gaussian_blur(img, sigma=2.0), img)


def test_blur_reduces_noise():
    rng = np.random.default_rng(0)
    img = 128.0 + rng.normal(0.0, 20.0, size=(30, 30))
    blurred = gaussian_blur(img, sigma=1.5)
    assert blurred.std() < img.std() * 0.5
    # Smoothing preserves the overall brightness.
    assert blurred.mean() == pytest.approx(img.mean(), abs=1.0)


def test_larger_sigma_smooths_more():
    img = make_test_image()
    small = gaussian_blur(img, sigma=0.5)
    large = gaussian_blur(img, sigma=3.0)
    assert large.std() < small.std()


def test_impulse_spreads_but_keeps_total():
    img = np.zeros((11, 11))
    img[5, 5] = 100.0
    blurred = gaussian_blur(img, sigma=1.0)
    assert blurred[5, 5] < 100.0
    assert blurred[5, 5] == blurred.max()
    assert blurred.sum() == pytest.approx(100.0, rel=1e-9)


def test_rgb_input_blurred_per_channel():
    rgb = np.zeros((10, 10, 3))
    rgb[:, 5:, :] = 255.0
    blurred = gaussian_blur(rgb, sigma=1.0)
    assert blurred.shape == rgb.shape
    assert blurred.dtype == np.float64
    # All three channels blurred identically for a gray-valued image.
    assert np.allclose(blurred[:, :, 0], blurred[:, :, 1])
    assert np.allclose(blurred[:, :, 1], blurred[:, :, 2])


def test_output_shape_and_dtype():
    img = np.random.default_rng(1).random((7, 11)) * 255
    out = gaussian_blur(img, sigma=1.0)
    assert out.shape == img.shape
    assert out.dtype == np.float64


def test_invalid_sigma_raises():
    img = np.zeros((5, 5))
    for bad in (0.0, -1.0, float("nan"), float("inf"), "1.0", True):
        with pytest.raises(ValueError):
            gaussian_blur(img, sigma=bad)


def test_invalid_kernel_size_raises():
    img = np.zeros((5, 5))
    for bad in (4, 0, -3, 2.5, "5"):
        with pytest.raises(ValueError):
            gaussian_blur(img, sigma=1.0, kernel_size=bad)


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((5,)))  # 1D
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        gaussian_blur(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        gaussian_blur(np.array([["a", "b"]]))  # non-numeric
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((5, 5, 4)))  # unsupported channels
