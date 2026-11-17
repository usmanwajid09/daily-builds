"""Tests for SLIC superpixel segmentation (NumPy only)."""

import numpy as np
import pytest

from superpixel_segmentation.superpixel_segmentation import (
    enforce_connectivity,
    slic,
)


def make_image(height=40, width=60, seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3))
    image[:, : width // 2] = [0.9, 0.1, 0.1]
    image[:, width // 2:] = [0.1, 0.1, 0.9]
    image += rng.normal(0, 0.02, image.shape)
    return np.clip(image, 0, 1)


def _is_connected(labels, label):
    """True if every pixel of `label` is reachable from the first via 4-neighbours."""
    mask = labels == label
    start = tuple(np.argwhere(mask)[0])
    seen = {start}
    stack = [start]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]
                    and mask[ny, nx] and (ny, nx) not in seen):
                seen.add((ny, nx))
                stack.append((ny, nx))
    return len(seen) == int(mask.sum())


def test_returns_correct_shapes_and_full_coverage():
    labels, centers = slic(make_image(), n_segments=12)
    assert labels.shape == (40, 60)
    assert labels.dtype.kind == "i"
    assert centers.ndim == 2 and centers.shape[1] == 5  # y, x, r, g, b
    assert set(np.unique(labels)) == set(range(len(np.unique(labels))))
    assert labels.size == 40 * 60


def test_segment_count_close_to_requested():
    labels, _ = slic(make_image(), n_segments=12)
    n = len(np.unique(labels))
    assert 6 <= n <= 12  # connectivity pass may merge a few tiny fragments


def test_grayscale_image_works():
    gray = np.tile(np.linspace(0, 1, 60), (40, 1))
    labels, centers = slic(gray, n_segments=8)
    assert labels.shape == (40, 60)
    assert centers.shape[1] == 3  # y, x, grey
    assert len(np.unique(labels)) >= 4


def test_uint8_image_works():
    image = (make_image() * 255).astype(np.uint8)
    labels, _ = slic(image, n_segments=10)
    assert labels.shape == (40, 60)
    assert len(np.unique(labels)) >= 4


def test_constant_image_does_not_crash():
    labels, centers = slic(np.full((20, 20, 3), 0.5), n_segments=4)
    assert labels.shape == (20, 20)
    assert len(np.unique(labels)) >= 1
    assert np.all(np.isfinite(centers))


def test_every_label_is_one_connected_region():
    labels, _ = slic(make_image(), n_segments=12)
    for lab in np.unique(labels):
        assert _is_connected(labels, lab), f"label {lab} is disconnected"


def test_deterministic_for_same_seed():
    image = make_image()
    labels1, centers1 = slic(image, n_segments=12, seed=0)
    labels2, centers2 = slic(image, n_segments=12, seed=0)
    assert np.array_equal(labels1, labels2)
    assert np.allclose(centers1, centers2)


def test_higher_compactness_gives_more_uniform_sizes():
    image = make_image()
    sizes_lo = np.bincount(slic(image, n_segments=12,
                                compactness=1.0)[0].ravel())
    sizes_hi = np.bincount(slic(image, n_segments=12,
                                compactness=40.0)[0].ravel())
    assert sizes_hi.std() <= sizes_lo.std() + 1.0  # allow small tolerance


def test_segments_follow_colour_boundary():
    # Left half red, right half blue: most superpixels should not straddle
    # the vertical boundary at x = 30.
    labels, _ = slic(make_image(), n_segments=12)
    straddling = 0
    for lab in np.unique(labels):
        ys, xs = np.where(labels == lab)
        if xs.min() < 30 < xs.max():
            straddling += 1
    assert straddling <= 3


def test_invalid_shape_1d():
    with pytest.raises(ValueError, match="2D"):
        slic(np.zeros(10))


def test_invalid_shape_4_channels():
    with pytest.raises(ValueError, match="channels"):
        slic(np.zeros((10, 10, 4)))


def test_float_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        slic(np.full((10, 10, 3), 1.5))


def test_n_segments_too_small():
    with pytest.raises(ValueError, match=">= 2"):
        slic(make_image(), n_segments=1)


def test_n_segments_more_than_pixels():
    with pytest.raises(ValueError, match="number of pixels"):
        slic(np.zeros((4, 4, 3)), n_segments=17)


def test_compactness_must_be_positive():
    with pytest.raises(ValueError, match="> 0"):
        slic(make_image(), n_segments=8, compactness=0)
    with pytest.raises(ValueError, match="> 0"):
        slic(make_image(), n_segments=8, compactness=-5.0)


def test_max_iter_must_be_positive():
    with pytest.raises(ValueError, match=">= 1"):
        slic(make_image(), n_segments=8, max_iter=0)


def test_enforce_connectivity_merges_fragment():
    labels = np.zeros((6, 6), dtype=int)
    labels[2:4, 2:4] = 1
    labels[0, 0] = 1  # isolated single pixel of label 1
    fixed = enforce_connectivity(labels, min_size=4)
    assert fixed[0, 0] == fixed[0, 1]  # fragment merged into neighbour
    assert fixed[2, 2] == fixed[3, 3]  # main block kept its own label
