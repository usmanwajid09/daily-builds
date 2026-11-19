import numpy as np
import pytest

from high_boost_filter.demo import make_test_image
from high_boost_filter.filter import (
    gaussian_kernel_1d,
    high_boost_filter,
    sharpening_mask,
)


def test_kernel_sums_to_one():
    kernel = gaussian_kernel_1d(1.0)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.allclose(kernel, kernel[::-1])


def test_constant_image_is_unchanged():
    img = np.full((10, 10), 128.0)
    assert np.allclose(high_boost_filter(img, k=2.0), img)
    assert np.allclose(sharpening_mask(img), 0.0)


def test_k_zero_returns_input_unchanged():
    img = make_test_image()
    assert np.allclose(high_boost_filter(img, k=0.0), img)


def test_edge_gets_steeper():
    img = make_test_image()
    sharp = high_boost_filter(img, k=2.0, sigma=1.0)
    mid = img.shape[1] // 2
    step_before = img[20, mid + 2] - img[20, mid - 3]
    step_after = sharp[20, mid + 2] - sharp[20, mid - 3]
    assert step_after > step_before


def test_bright_side_rises_dark_side_falls():
    img = make_test_image()
    sharp = high_boost_filter(img, k=1.0, sigma=1.0)
    mid = img.shape[1] // 2
    assert sharp[20, mid + 1] > img[20, mid + 1]
    assert sharp[20, mid - 1] < img[20, mid - 1]


def test_larger_k_sharpens_more():
    img = make_test_image()
    mild = high_boost_filter(img, k=0.5, sigma=1.0)
    strong = high_boost_filter(img, k=3.0, sigma=1.0)
    mid = img.shape[1] // 2
    step_mild = mild[20, mid + 2] - mild[20, mid - 3]
    step_strong = strong[20, mid + 2] - strong[20, mid - 3]
    assert step_strong > step_mild


def test_output_stays_in_range_and_keeps_shape():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=(25, 30))
    out = high_boost_filter(img, k=5.0, sigma=1.0)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_mask_is_zero_on_flat_areas():
    img = np.full((12, 12), 90.0)
    img[5:7, 5:7] = 200.0
    mask = sharpening_mask(img, sigma=1.0)
    assert mask[0, 0] == pytest.approx(0.0)
    assert mask[5, 5] > 0  # bright patch stands out from its blur


def test_rgb_image_works_per_channel():
    img = np.zeros((12, 12, 3))
    img[:, 6:, 0] = 200.0   # red step on the right
    img[:, :6, 1] = 150.0   # green step on the left
    out = high_boost_filter(img, k=2.0, sigma=1.0)
    assert out.shape == img.shape
    assert out[6, 6, 0] > img[6, 6, 0]   # just right of the red step
    assert out[6, 5, 1] > img[6, 5, 1]   # just left of the green step
    assert np.allclose(out[:, :, 2], 0.0)  # blue channel is flat


def test_invalid_k_raises():
    img = np.zeros((8, 8))
    with pytest.raises(ValueError):
        high_boost_filter(img, k=-1.0)
    with pytest.raises(ValueError):
        high_boost_filter(img, k=float("inf"))


def test_invalid_sigma_raises():
    img = np.zeros((8, 8))
    with pytest.raises(ValueError):
        high_boost_filter(img, sigma=0.0)
    with pytest.raises(ValueError):
        high_boost_filter(img, sigma=-2.0)
    with pytest.raises(ValueError):
        sharpening_mask(img, sigma=0.0)


def test_bad_image_raises():
    with pytest.raises(ValueError):
        high_boost_filter(np.zeros((0, 5)))
    with pytest.raises(ValueError):
        high_boost_filter(np.zeros((5, 5, 4)))
    with pytest.raises(ValueError):
        high_boost_filter(np.zeros((5, 5, 3, 3)))
    with pytest.raises(ValueError):
        high_boost_filter(np.full((5, 5), np.inf))
