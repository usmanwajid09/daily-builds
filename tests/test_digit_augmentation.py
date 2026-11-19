import numpy as np
import pytest

from digit_augmentation import (
    add_gaussian_noise,
    augment,
    augment_n,
    flip_horizontal,
    flip_vertical,
    rotate90,
    translate,
)
from digit_augmentation.demo import make_digit


def _image():
    """A small 4x5 test image with a single bright pixel."""
    image = np.zeros((4, 5), dtype=np.float64)
    image[1, 2] = 1.0
    return image


def test_rotate90_matches_np_rot90():
    image = _image()
    assert np.array_equal(rotate90(image, k=1), np.rot90(image, k=1))
    assert np.array_equal(rotate90(image, k=2), np.rot90(image, k=2))


def test_four_rotations_return_original():
    image = _image()
    assert np.array_equal(rotate90(image, k=4), image)


def test_flip_horizontal_matches_np_fliplr():
    image = _image()
    assert np.array_equal(flip_horizontal(image), np.fliplr(image))


def test_flip_vertical_matches_np_flipud():
    image = _image()
    assert np.array_equal(flip_vertical(image), np.flipud(image))


def test_translate_zero_is_identity():
    image = _image()
    assert np.array_equal(translate(image, dy=0, dx=0), image)


def test_translate_shifts_bright_pixel():
    image = _image()
    shifted = translate(image, dy=1, dx=-1)
    assert shifted.shape == image.shape
    assert shifted[2, 1] == 1.0
    assert shifted[1, 2] == 0.0
    assert shifted.sum() == image.sum()


def test_translate_fills_rolled_in_area_with_zero():
    image = np.ones((3, 3), dtype=np.float64)
    shifted = translate(image, dy=1, dx=0)
    assert (shifted[0, :] == 0).all()
    assert (shifted[1:, :] == 1).all()


def test_noise_std_zero_is_identity():
    image = make_digit()
    assert np.array_equal(add_gaussian_noise(image, std=0.0), image)


def test_noise_with_fixed_seed_is_reproducible():
    image = make_digit()
    first = add_gaussian_noise(image, std=0.1, seed=42)
    second = add_gaussian_noise(image, std=0.1, seed=42)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)


def test_augment_preserves_shape():
    image = make_digit()
    out = augment(image, seed=1)
    assert out.shape == image.shape


def test_augment_n_returns_n_images_of_right_shape():
    image = make_digit()
    copies = augment_n(image, n=5, seed=42)
    assert len(copies) == 5
    assert all(copy.shape == image.shape for copy in copies)


def test_augment_n_with_seed_is_reproducible():
    image = make_digit()
    first = augment_n(image, n=3, seed=7)
    second = augment_n(image, n=3, seed=7)
    assert all(
        np.array_equal(a, b) for a, b in zip(first, second)
    )
    assert not all(
        np.array_equal(first[0], copy) for copy in first[1:]
    )


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        rotate90(np.zeros((4, 4, 3)))          # 3D array
    with pytest.raises(ValueError):
        flip_horizontal(np.zeros((0, 0)))      # empty
    with pytest.raises(ValueError):
        augment_n(make_digit(), n=0)           # non-positive n
    with pytest.raises(ValueError):
        augment_n(make_digit(), n=-3)          # negative n
    with pytest.raises(ValueError):
        add_gaussian_noise(make_digit(), std=-0.1)  # negative std
    with pytest.raises(ValueError):
        rotate90(make_digit(), k=1.5)          # non-int k
