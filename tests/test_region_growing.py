"""Tests for region_growing (region growing segmentation)."""

import numpy as np
import pytest

from region_growing.region_growing import grow_from_seed, region_grow


def _two_discs():
    yy, xx = np.mgrid[0:40, 0:60]
    image = np.zeros((40, 60), dtype=np.float64)
    image[(yy - 12) ** 2 + (xx - 15) ** 2 <= 8 ** 2] = 200.0
    image[(yy - 28) ** 2 + (xx - 45) ** 2 <= 6 ** 2] = 120.0
    return image


def test_grows_whole_disc():
    image = _two_discs()
    mask = grow_from_seed(image, (12, 15), tolerance=10.0)
    # Disc radius 8 -> area pi*64 ~= 201 pixels.
    assert 190 <= mask.sum() <= 210
    assert mask[12, 15]
    assert not mask[28, 45]


def test_tolerance_zero_grows_only_equal_pixels():
    image = _two_discs()
    mask = grow_from_seed(image, (12, 15), tolerance=0.0)
    assert 190 <= mask.sum() <= 210
    assert not mask[0, 0]  # background (0) differs from seed (200)


def test_constant_image_grows_everywhere():
    image = np.full((8, 8), 42.0)
    mask = grow_from_seed(image, (3, 3), tolerance=0.0)
    assert mask.all()


def test_large_tolerance_grows_whole_image():
    image = _two_discs()
    mask = grow_from_seed(image, (12, 15), tolerance=250.0)
    assert mask.all()


def test_barrier_blocks_growth():
    # A bright wall splits the image; the seed's region cannot cross it.
    image = np.full((10, 10), 50.0)
    image[:, 5] = 200.0
    mask = grow_from_seed(image, (5, 2), tolerance=1.0)
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()


def test_8_connectivity_crosses_diagonal():
    image = np.zeros((3, 3))
    image[0, 0] = image[1, 1] = image[2, 2] = 100.0
    assert grow_from_seed(image, (0, 0), 0.0, connectivity=4).sum() == 1
    assert grow_from_seed(image, (0, 0), 0.0, connectivity=8).sum() == 3


def test_multiple_seeds_union():
    image = _two_discs()
    mask = region_grow(image, [(12, 15), (28, 45)], tolerance=10.0)
    assert mask[12, 15] and mask[28, 45]
    assert 190 + 100 <= mask.sum() <= 210 + 120


def test_output_is_bool_mask_same_shape():
    image = _two_discs()
    mask = region_grow(image, [(12, 15)], tolerance=10.0)
    assert mask.dtype == bool
    assert mask.shape == image.shape


def test_uint8_input_accepted():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:5, 2:5] = 200
    mask = grow_from_seed(image, (3, 3), tolerance=0.0)
    assert mask[2:5, 2:5].all()
    assert mask.sum() == 9


def test_rejects_1d_image():
    with pytest.raises(ValueError, match="2D"):
        grow_from_seed(np.zeros(10), (0,), tolerance=1.0)


def test_rejects_rgb_image():
    with pytest.raises(ValueError, match="2D"):
        grow_from_seed(np.zeros((4, 4, 3)), (0, 0), tolerance=1.0)


def test_rejects_empty_image():
    with pytest.raises(ValueError, match="empty"):
        grow_from_seed(np.zeros((0, 0)), (0, 0), tolerance=1.0)


def test_rejects_nan_image():
    image = np.zeros((5, 5))
    image[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        grow_from_seed(image, (0, 0), tolerance=1.0)


def test_rejects_empty_seeds():
    with pytest.raises(ValueError, match="empty"):
        region_grow(np.zeros((5, 5)), [], tolerance=1.0)


def test_rejects_seed_out_of_bounds():
    image = np.zeros((5, 5))
    with pytest.raises(ValueError, match="outside"):
        grow_from_seed(image, (5, 0), tolerance=1.0)
    with pytest.raises(ValueError, match="outside"):
        grow_from_seed(image, (0, -1), tolerance=1.0)


def test_rejects_non_integer_seed():
    with pytest.raises(ValueError, match="integers"):
        grow_from_seed(np.zeros((5, 5)), (1.5, 0), tolerance=1.0)


def test_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="non-negative"):
        grow_from_seed(np.zeros((5, 5)), (0, 0), tolerance=-1.0)


def test_rejects_bad_connectivity():
    with pytest.raises(ValueError, match="connectivity"):
        grow_from_seed(np.zeros((5, 5)), (0, 0), tolerance=1.0, connectivity=6)
