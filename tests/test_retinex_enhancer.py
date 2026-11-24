import numpy as np
import pytest

from retinex_enhancer.retinex import (
    enhance,
    gaussian_blur,
    gaussian_kernel_1d,
    multi_scale_retinex,
    multi_scale_retinex_cr,
    single_scale_retinex,
)
from retinex_enhancer.demo import make_test_image


# --- gaussian_kernel_1d / gaussian_blur -----------------------------------


def test_gaussian_kernel_sums_to_one_and_is_symmetric():
    for sigma in (0.5, 1.0, 2.5):
        kernel = gaussian_kernel_1d(sigma)
        assert kernel.sum() == pytest.approx(1.0)
        assert np.allclose(kernel, kernel[::-1])
        assert kernel[len(kernel) // 2] == kernel.max()


def test_gaussian_kernel_size_covers_three_sigma():
    assert len(gaussian_kernel_1d(1.0)) == 7   # 2*ceil(3)+1
    assert len(gaussian_kernel_1d(2.0)) == 13  # 2*ceil(6)+1


def test_gaussian_kernel_invalid_sigma_raises():
    with pytest.raises(ValueError, match="positive"):
        gaussian_kernel_1d(0)
    with pytest.raises(ValueError, match="positive"):
        gaussian_kernel_1d(-1.0)


def test_gaussian_blur_preserves_flat_image():
    image = np.full((20, 20), 100.0)
    assert gaussian_blur(image, 2.0) == pytest.approx(image, abs=1e-9)


def test_gaussian_blur_preserves_shape_grayscale_and_rgb():
    gray = np.random.default_rng(0).random((15, 20)) * 255
    rgb = np.random.default_rng(0).random((15, 20, 3)) * 255
    assert gaussian_blur(gray, 1.5).shape == gray.shape
    assert gaussian_blur(rgb, 1.5).shape == rgb.shape


def test_gaussian_blur_smooths_a_spike():
    image = np.zeros((21, 21))
    image[10, 10] = 255.0
    blurred = gaussian_blur(image, 2.0)
    assert blurred[10, 10] < 255.0
    assert blurred[10, 10] > blurred[10, 13]
    assert blurred.sum() == pytest.approx(255.0, rel=1e-6)


# --- single_scale_retinex -------------------------------------------------


def test_ssr_flat_image_maps_to_mid_grey():
    image = np.full((10, 10), 80.0)
    result = single_scale_retinex(image, sigma=5.0)
    assert np.all(result == 128.0)


def test_ssr_output_range_and_shape():
    image = make_test_image()
    result = single_scale_retinex(image, sigma=15.0)
    assert result.shape == image.shape
    assert result.min() >= 0.0
    assert result.max() <= 255.0


def test_ssr_brightens_dark_image():
    image = make_test_image()
    result = single_scale_retinex(image, sigma=15.0)
    assert result.mean() > image.mean()


# --- multi_scale_retinex --------------------------------------------------


def test_msr_single_sigma_matches_ssr():
    image = make_test_image()
    msr = multi_scale_retinex(image, sigmas=(80.0,))
    ssr = single_scale_retinex(image, sigma=80.0)
    assert msr == pytest.approx(ssr, abs=1e-9)


def test_msr_output_range_and_shape():
    image = make_test_image()
    result = multi_scale_retinex(image)
    assert result.shape == image.shape
    assert result.min() >= 0.0
    assert result.max() <= 255.0


def test_msr_brightens_dark_image_and_keeps_detail():
    image = make_test_image()
    result = multi_scale_retinex(image)
    assert result.mean() > image.mean()
    assert result.std() > 0


def test_msr_empty_sigmas_raises():
    with pytest.raises(ValueError, match="non-empty"):
        multi_scale_retinex(make_test_image(), sigmas=())


def test_msr_bad_sigma_raises():
    with pytest.raises(ValueError, match="positive"):
        multi_scale_retinex(make_test_image(), sigmas=(15.0, 0.0))


# --- multi_scale_retinex_cr / enhance -------------------------------------


def test_msrcr_output_range_and_shape_rgb():
    image = make_test_image()
    result = multi_scale_retinex_cr(image)
    assert result.shape == image.shape
    assert result.min() >= 0.0
    assert result.max() <= 255.0


def test_msrcr_grayscale_matches_msr_shape():
    gray = make_test_image()[:, :, 0]
    result = multi_scale_retinex_cr(gray)
    assert result.shape == gray.shape
    assert result.min() >= 0.0 and result.max() <= 255.0


def test_msrcr_invalid_gain_and_alpha_raise():
    image = make_test_image()
    with pytest.raises(ValueError, match="positive"):
        multi_scale_retinex_cr(image, alpha=0.0)
    with pytest.raises(ValueError, match="positive"):
        multi_scale_retinex_cr(image, gain=0.0)


def test_enhance_returns_uint8_in_range():
    for method in ("ssr", "msr", "msrcr"):
        result = enhance(make_test_image(), method=method)
        assert result.dtype == np.uint8
        assert result.min() >= 0 and result.max() <= 255


def test_enhance_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        enhance(make_test_image(), method="clahe")


def test_invalid_images_raise():
    with pytest.raises(ValueError):
        single_scale_retinex(np.full((5, 5), -1.0))
    with pytest.raises(ValueError):
        single_scale_retinex(np.full((5, 5, 4), 10.0))
    with pytest.raises(ValueError):
        single_scale_retinex(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        single_scale_retinex(None)
