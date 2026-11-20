"""Tests for gabor_filter."""

import numpy as np
import pytest

from gabor_filter import (
    apply_gabor_bank,
    gabor_bank,
    gabor_kernel,
    gabor_responses,
    texture_energy,
)
from gabor_filter.demo import make_test_image


def test_kernel_shape_and_dtype():
    kernel = gabor_kernel(ksize=21, sigma=4.0, theta=0.0, wavelength=8.0)
    assert kernel.shape == (21, 21)
    assert kernel.dtype == np.float64
    assert np.isfinite(kernel).all()


def test_kernel_is_zero_mean():
    kernel = gabor_kernel(ksize=15, sigma=3.0, theta=np.pi / 4,
                          wavelength=6.0)
    assert abs(kernel.mean()) < 1e-12


def test_kernel_orientation_changes_kernel():
    k0 = gabor_kernel(ksize=15, sigma=3.0, theta=0.0, wavelength=6.0)
    k90 = gabor_kernel(ksize=15, sigma=3.0, theta=np.pi / 2, wavelength=6.0)
    assert not np.allclose(k0, k90)
    # A 90-degree rotation of the kernel equals the 0-degree kernel
    # transposed (up to the carrier phase).
    assert np.allclose(np.rot90(k0), k90, atol=1e-10)


def test_kernel_invalid_ksize():
    with pytest.raises(ValueError):
        gabor_kernel(ksize=20, sigma=4.0, theta=0.0, wavelength=8.0)
    with pytest.raises(ValueError):
        gabor_kernel(ksize=1, sigma=4.0, theta=0.0, wavelength=8.0)


def test_kernel_invalid_params():
    with pytest.raises(ValueError):
        gabor_kernel(ksize=15, sigma=0.0, theta=0.0, wavelength=8.0)
    with pytest.raises(ValueError):
        gabor_kernel(ksize=15, sigma=4.0, theta=0.0, wavelength=0.0)
    with pytest.raises(ValueError):
        gabor_kernel(ksize=15, sigma=4.0, theta=0.0, wavelength=8.0,
                     gamma=-1.0)


def test_bank_count_and_order():
    kernels = gabor_bank(n_orientations=4, wavelengths=(4.0, 8.0, 16.0))
    assert len(kernels) == 12
    assert all(k.shape == (21, 21) for k in kernels)


def test_bank_invalid_params():
    with pytest.raises(ValueError):
        gabor_bank(n_orientations=0)
    with pytest.raises(ValueError):
        gabor_bank(wavelengths=())
    with pytest.raises(ValueError):
        gabor_bank(wavelengths=(-4.0,))


def test_apply_bank_shape_and_dtype():
    image = make_test_image()
    kernels = gabor_bank(n_orientations=4, wavelengths=(8.0,))
    responses = apply_gabor_bank(image, kernels)
    assert responses.shape == (64, 64, 4)
    assert responses.dtype == np.float64
    assert np.isfinite(responses).all()


def test_constant_image_gives_zero_response():
    image = np.full((20, 20), 100.0)
    kernels = gabor_bank(n_orientations=2, wavelengths=(8.0,))
    responses = apply_gabor_bank(image, kernels)
    assert np.allclose(responses, 0.0, atol=1e-10)


def test_orientation_selectivity():
    # Theta=0 filter (vertical stripes) fires on the left half, not
    # on the horizontal stripes on the right half.
    image = make_test_image()
    kernels = gabor_bank(n_orientations=4, wavelengths=(8.0,))
    responses = apply_gabor_bank(image, kernels)
    left = np.abs(responses[:, :32, 0]).mean()
    right = np.abs(responses[:, 32:, 0]).mean()
    assert left > 10 * right
    # Theta=90 filter shows the mirror behaviour.
    left90 = np.abs(responses[:, :32, 2]).mean()
    right90 = np.abs(responses[:, 32:, 2]).mean()
    assert right90 > 10 * left90


def test_apply_bank_rejects_bad_input():
    kernels = gabor_bank(n_orientations=2, wavelengths=(8.0,))
    with pytest.raises(ValueError):
        apply_gabor_bank(np.zeros((10, 10, 3)), kernels)
    with pytest.raises(ValueError):
        apply_gabor_bank(np.zeros((10, 10)), [])
    with pytest.raises(ValueError):
        apply_gabor_bank(np.zeros((0, 0)), kernels)


def test_gabor_responses_wrapper():
    image = make_test_image()
    responses = gabor_responses(image, n_orientations=4,
                                wavelengths=(8.0, 16.0))
    assert responses.shape == (64, 64, 8)


def test_texture_energy_shape_and_sign():
    image = make_test_image()
    responses = gabor_responses(image, n_orientations=4,
                                wavelengths=(8.0,))
    energy = texture_energy(responses, window=9)
    assert energy.shape == responses.shape
    assert np.all(energy >= 0.0)
    # Energy is high where stripes are, low on a flat image.
    flat = texture_energy(apply_gabor_bank(np.full((20, 20), 50.0),
                                           gabor_bank(2, (8.0,))))
    assert np.allclose(flat, 0.0, atol=1e-10)


def test_texture_energy_invalid_window():
    responses = np.zeros((10, 10, 2))
    with pytest.raises(ValueError):
        texture_energy(responses, window=8)
    with pytest.raises(ValueError):
        texture_energy(np.zeros((10, 10)), window=9)
