import numpy as np
import pytest

from median_filter.demo import make_test_image
from median_filter.filter import median_filter, salt_and_pepper_pixel_count


def test_constant_image_is_unchanged():
    img = np.full((8, 8), 128.0)
    assert np.allclose(median_filter(img, kernel_size=3), img)
    assert np.allclose(median_filter(img, kernel_size=5), img)


def test_single_impulse_is_removed():
    img = np.zeros((5, 5))
    img[2, 2] = 255.0  # one salt spike in a black field
    assert np.allclose(median_filter(img, kernel_size=3), 0.0)
    img = np.full((5, 5), 255.0)
    img[2, 2] = 0.0  # one pepper spike in a white field
    assert np.allclose(median_filter(img, kernel_size=3), 255.0)


def test_kernel_size_one_is_identity():
    img = np.arange(25, dtype=np.float64).reshape(5, 5)
    assert np.array_equal(median_filter(img, kernel_size=1), img)


def test_step_edge_survives():
    img = np.zeros((10, 10))
    img[:, 5:] = 200.0
    out = median_filter(img, kernel_size=3)
    assert np.allclose(out[:, :4], 0.0)
    assert np.allclose(out[:, 6:], 200.0)


def test_salt_and_pepper_noise_is_reduced():
    clean, noisy = make_test_image()
    denoised = median_filter(noisy, kernel_size=3)
    assert salt_and_pepper_pixel_count(denoised) < salt_and_pepper_pixel_count(noisy)
    mae_before = float(np.mean(np.abs(noisy - clean)))
    mae_after = float(np.mean(np.abs(denoised - clean)))
    assert mae_after < mae_before * 0.5


def test_larger_kernel_removes_denser_noise():
    rng = np.random.default_rng(1)
    clean = np.full((15, 15), 120.0)
    noisy = clean.copy()
    flip = rng.random(size=clean.shape) < 0.25
    noisy[flip] = 255.0
    small = median_filter(noisy, kernel_size=3)
    large = median_filter(noisy, kernel_size=5)
    assert float(np.mean(np.abs(large - clean))) <= float(
        np.mean(np.abs(small - clean)))


def test_rgb_image_filtered_per_channel():
    img = np.zeros((7, 7, 3))
    img[:, :, 0] = 100.0
    img[3, 3, 0] = 255.0  # salt spike in the red channel only
    out = median_filter(img, kernel_size=3)
    assert out.shape == img.shape
    assert out[3, 3, 0] == pytest.approx(100.0)
    assert np.allclose(out[:, :, 1:], 0.0)


def test_output_shape_and_dtype():
    img = np.arange(36, dtype=np.uint8).reshape(6, 6)
    out = median_filter(img, kernel_size=3)
    assert out.shape == img.shape
    assert out.dtype == np.float64


def test_even_kernel_size_raises():
    img = np.zeros((8, 8))
    with pytest.raises(ValueError):
        median_filter(img, kernel_size=2)
    with pytest.raises(ValueError):
        median_filter(img, kernel_size=4)


def test_bad_kernel_size_raises():
    img = np.zeros((8, 8))
    with pytest.raises(ValueError):
        median_filter(img, kernel_size=0)
    with pytest.raises(ValueError):
        median_filter(img, kernel_size=-3)
    with pytest.raises(ValueError):
        median_filter(img, kernel_size=3.0)
    with pytest.raises(ValueError):
        median_filter(img, kernel_size="3")


def test_bad_image_raises():
    with pytest.raises(ValueError):
        median_filter(np.zeros((0, 5)))
    with pytest.raises(ValueError):
        median_filter(np.zeros((5, 5, 4)))
    with pytest.raises(ValueError):
        median_filter(np.zeros((5,)))
    with pytest.raises(ValueError):
        median_filter(np.full((5, 5), np.nan))


def test_salt_and_pepper_pixel_count():
    img = np.zeros((4, 4))
    img[0, 0] = 255.0
    img[1, 1] = 255.0
    img[2, 2] = 0.0
    assert salt_and_pepper_pixel_count(img) == 16  # all pixels are 0 or 255
    assert salt_and_pepper_pixel_count(np.full((4, 4), 128.0)) == 0


if __name__ == "__main__":
    raise SystemExit("run with: python -m pytest tests/test_median_filter.py -v")
