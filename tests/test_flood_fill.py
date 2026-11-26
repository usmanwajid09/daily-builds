"""Tests for flood_fill (flood_fill/model.py)."""

import numpy as np
import pytest

from flood_fill.model import fill_holes, flood_fill, region_grow
from flood_fill.demo import make_image, make_ring


def test_uniform_region_fills_entire_image():
    image = np.full((5, 6), 42, dtype=np.uint8)
    mask = flood_fill(image, (2, 3))
    assert mask.dtype == bool
    assert mask.shape == image.shape
    assert mask.all()


def test_fill_stops_at_different_valued_region():
    image = np.zeros((5, 5), dtype=np.uint8)
    image[:, 3:] = 200
    mask = flood_fill(image, (2, 1), tolerance=0.0)
    assert mask[:, :3].all()
    assert not mask[:, 3:].any()


def test_small_tolerance_excludes_neighbour():
    image = np.array([[100, 110],
                      [100, 100]], dtype=np.uint8)
    mask = flood_fill(image, (0, 0), tolerance=5.0)
    assert mask[0, 0] and mask[1, 0] and mask[1, 1]
    assert not mask[0, 1]


def test_large_tolerance_includes_neighbour():
    image = np.array([[100, 110],
                      [100, 100]], dtype=np.uint8)
    mask = flood_fill(image, (0, 0), tolerance=10.0)
    assert mask.all()


def test_connectivity_4_vs_8_diagonal_only():
    # Only diagonal links between the two 100-valued pixels.
    image = np.array([[100, 0],
                      [0, 100]], dtype=np.uint8)
    assert flood_fill(image, (0, 0), connectivity=4).sum() == 1
    assert flood_fill(image, (0, 0), connectivity=8).sum() == 2


def test_seed_out_of_bounds_raises():
    image = np.zeros((4, 4), dtype=np.uint8)
    for seed in [(4, 0), (0, 4), (-1, 0), (0, -1)]:
        with pytest.raises(ValueError, match="outside the image"):
            flood_fill(image, seed)


def test_bad_connectivity_raises():
    image = np.zeros((3, 3), dtype=np.uint8)
    for connectivity in (2, 6, 12):
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            flood_fill(image, (1, 1), connectivity=connectivity)


def test_negative_tolerance_raises():
    image = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="non-negative"):
        flood_fill(image, (1, 1), tolerance=-1.0)


def test_non_2d_input_raises():
    with pytest.raises(ValueError, match="must be 2D"):
        flood_fill(np.zeros((3, 3, 3), dtype=np.uint8), (1, 1))
    with pytest.raises(ValueError, match="must be 2D"):
        flood_fill(np.zeros(5, dtype=np.uint8), (1,))


def test_non_numeric_dtype_raises():
    with pytest.raises(TypeError, match="numeric dtype"):
        flood_fill(np.zeros((3, 3), dtype=bool), (1, 1))
    with pytest.raises(TypeError, match="must be a numpy array"):
        flood_fill([[0, 0], [0, 0]], (0, 0))


def test_fill_holes_fills_enclosed_hole():
    ring = make_ring()
    assert not ring[4, 4]
    filled = fill_holes(ring)
    assert filled.all()
    assert int(filled.sum()) == 81


def test_fill_holes_leaves_border_background_alone():
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True  # solid block, no hole inside
    filled = fill_holes(mask)
    assert (filled == mask).all()
    # A hole touching the border is not enclosed: leave it.
    mask2 = np.ones((5, 5), dtype=bool)
    mask2[0, 2] = False  # notch open to the border
    assert not fill_holes(mask2)[0, 2]


def test_region_grow_agrees_with_flood_fill():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 3, size=(10, 12)).astype(np.uint8) * 50
    for tolerance, connectivity in ((0.0, 4), (25.0, 8), (50.0, 4)):
        expected = flood_fill(image, (5, 5), tolerance, connectivity)
        got = region_grow(image, (5, 5), tolerance, connectivity)
        assert (got == expected).all()


def test_demo_disc_fill_matches_true_area():
    image, disc = make_image()
    mask = flood_fill(image, (30, 25), tolerance=5.0, connectivity=4)
    assert int(mask.sum()) == int(disc.sum())
    assert (mask == disc).all()
    # The 180-valued rectangle must not be reached.
    assert not mask[10:25, 55:75].any()


def test_float_image_and_tolerance():
    image = np.array([[1.0, 1.05, 2.0],
                      [1.0, 1.0, 2.0]], dtype=float)
    mask = flood_fill(image, (0, 0), tolerance=0.1)
    assert mask[0, 0] and mask[0, 1] and mask[1, 0] and mask[1, 1]
    assert not mask[0, 2] and not mask[1, 2]
