import numpy as np
import pytest

from laplacian_pyramid.pyramid import (
    blend_images,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct,
)
from laplacian_pyramid.demo import make_test_image


def test_gaussian_pyramid_level_shapes_halve_each_level():
    image = make_test_image(64)
    pyramid = build_gaussian_pyramid(image, levels=4)
    assert [level.shape for level in pyramid] == [
        (64, 64),
        (32, 32),
        (16, 16),
        (8, 8),
    ]


def test_laplacian_pyramid_has_requested_number_of_levels():
    image = make_test_image(64)
    for levels in (1, 3, 5):
        pyramid = build_laplacian_pyramid(image, levels=levels)
        assert len(pyramid) == levels
        # Top level is the full-size detail level.
        assert pyramid[0].shape == image.shape


def test_reconstruction_error_is_tiny_on_float_image():
    image = make_test_image(64)
    pyramid = build_laplacian_pyramid(image, levels=4)
    rebuilt = reconstruct(pyramid)
    assert rebuilt.shape == image.shape
    assert np.max(np.abs(rebuilt - image)) < 1.0
    assert np.mean((rebuilt - image) ** 2) < 1e-10


def test_reconstruct_round_trips_random_image():
    rng = np.random.default_rng(0)
    image = rng.random((48, 40)) * 255.0
    rebuilt = reconstruct(build_laplacian_pyramid(image, levels=4))
    assert rebuilt.shape == image.shape
    assert np.allclose(rebuilt, image, atol=1e-9)


def test_reconstruction_error_is_tiny_on_uint8_image():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    rebuilt = reconstruct(build_laplacian_pyramid(image, levels=3))
    assert np.max(np.abs(rebuilt - image.astype(np.float64))) < 2.0


def test_blend_output_shape_and_range():
    image_a = make_test_image(64)
    image_b = np.flipud(make_test_image(64)) * 0.4
    mask = np.zeros((64, 64))
    mask[:, :32] = 1.0
    blended = blend_images(image_a, image_b, mask, levels=4)
    assert blended.shape == image_a.shape
    assert blended.min() >= min(image_a.min(), image_b.min()) - 1e-9
    assert blended.max() <= max(image_a.max(), image_b.max()) + 1e-9


def test_blend_with_all_ones_mask_returns_image_a():
    rng = np.random.default_rng(2)
    image_a = rng.random((32, 32)) * 200.0
    image_b = rng.random((32, 32)) * 200.0
    blended = blend_images(image_a, image_b, np.ones((32, 32)), levels=3)
    assert np.allclose(blended, image_a, atol=1.0)


def test_blend_with_all_zeros_mask_returns_image_b():
    rng = np.random.default_rng(3)
    image_a = rng.random((32, 32)) * 200.0
    image_b = rng.random((32, 32)) * 200.0
    blended = blend_images(image_a, image_b, np.zeros((32, 32)), levels=3)
    assert np.allclose(blended, image_b, atol=1.0)


def test_invalid_levels_raise_value_error():
    image = make_test_image(32)
    with pytest.raises(ValueError):
        build_laplacian_pyramid(image, levels=0)
    with pytest.raises(ValueError):
        build_laplacian_pyramid(image, levels=-2)
    with pytest.raises(ValueError):
        # 32x32 supports at most 6 levels (32, 16, 8, 4, 2, 1).
        build_laplacian_pyramid(image, levels=7)
    with pytest.raises(ValueError):
        build_gaussian_pyramid(image, levels=0)


def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError):
        build_laplacian_pyramid(np.zeros(10), levels=2)  # 1D
    with pytest.raises(ValueError):
        build_laplacian_pyramid(np.zeros((0, 10)), levels=2)  # empty
    with pytest.raises(ValueError):
        build_laplacian_pyramid(np.zeros((10, 0)), levels=2)  # empty
    with pytest.raises(ValueError):
        build_laplacian_pyramid(np.zeros((2, 2, 2, 2)), levels=2)  # 4D


def test_rgb_image_works_and_shapes_preserved():
    rng = np.random.default_rng(4)
    image = rng.random((32, 32, 3)) * 255.0
    gaussian = build_gaussian_pyramid(image, levels=3)
    assert [level.shape for level in gaussian] == [
        (32, 32, 3),
        (16, 16, 3),
        (8, 8, 3),
    ]
    rebuilt = reconstruct(build_laplacian_pyramid(image, levels=3))
    assert rebuilt.shape == image.shape
    assert np.max(np.abs(rebuilt - image)) < 1.0
    # RGB blend keeps the channel dimension too.
    mask = np.zeros((32, 32))
    mask[:, :16] = 1.0
    blended = blend_images(image, np.flip(image, axis=1), mask, levels=3)
    assert blended.shape == image.shape
