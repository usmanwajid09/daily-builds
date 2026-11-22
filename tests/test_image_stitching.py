"""Tests for image_stitching."""

import numpy as np
import pytest

from image_stitching import (
    find_best_shift,
    normalized_cross_correlation,
    stitch_images,
)


def make_pair(shape=(40, 48), shift=(4, 7), seed=0):
    """Two views of one random scene; img2 sits at ``shift`` vs img1."""
    rng = np.random.default_rng(seed)
    dy, dx = shift
    h, w = shape
    base = rng.random((h + abs(dy), w + abs(dx))) * 255.0
    img1 = base[max(0, -dy):max(0, -dy) + h, max(0, -dx):max(0, -dx) + w]
    img2 = base[max(0, dy):max(0, dy) + h, max(0, dx):max(0, dx) + w]
    return img1.copy(), img2.copy()


def test_positive_shift_recovered_exactly():
    img1, img2 = make_pair(shift=(4, 7))
    found, score = find_best_shift(img1, img2, max_shift=12)
    assert found == (4, 7)
    assert score == pytest.approx(1.0)


def test_negative_shift_recovered_exactly():
    img1, img2 = make_pair(shift=(-6, -9), seed=1)
    found, score = find_best_shift(img1, img2, max_shift=12)
    assert found == (-6, -9)
    assert score == pytest.approx(1.0)


def test_zero_shift_recovered():
    img1, img2 = make_pair(shift=(0, 0), seed=2)
    found, score = find_best_shift(img1, img2, max_shift=8)
    assert found == (0, 0)
    assert score == pytest.approx(1.0)


def test_mixed_sign_shift_recovered():
    img1, img2 = make_pair(shift=(5, -11), seed=3)
    found, _ = find_best_shift(img1, img2, max_shift=14)
    assert found == (5, -11)


def test_ncc_identical_is_one_and_unrelated_is_lower():
    rng = np.random.default_rng(4)
    patch = rng.random((12, 12)) * 255.0
    assert normalized_cross_correlation(patch, patch) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    a = rng.random((12, 12)) * 255.0
    b = rng.random((12, 12)) * 255.0
    score = normalized_cross_correlation(a, b)
    assert score < 0.5
    assert normalized_cross_correlation(a, a) > score


def test_stitch_shape_is_union():
    img1, img2 = make_pair(shape=(40, 48), shift=(4, 7))
    pano = stitch_images(img1, img2, shift=(4, 7))
    assert pano.shape == (44, 55)


def test_stitch_non_overlap_region_exact():
    img1, img2 = make_pair(shape=(40, 48), shift=(4, 7))
    pano = stitch_images(img1, img2, shift=(4, 7))
    # Top rows and left columns are covered only by img1.
    assert np.array_equal(pano[0:4, 0:48], img1[0:4, :])
    assert np.array_equal(pano[4:40, 0:7], img1[4:40, 0:7])
    # Bottom-right corner is covered only by img2.
    assert np.array_equal(pano[40:44, 48:55], img2[36:40, 41:48])


def test_stitch_overlap_blend_is_between_sources():
    img1, img2 = make_pair(shape=(40, 48), shift=(4, 7))
    pano = stitch_images(img1, img2, shift=(4, 7))
    # Overlap in canvas coords: rows 4-39, cols 7-47.
    overlap = pano[4:40, 7:48]
    lo = np.minimum(img1[4:40, 7:48], img2[0:36, 0:41])
    hi = np.maximum(img1[4:40, 7:48], img2[0:36, 0:41])
    assert np.all(overlap >= lo - 1e-9)
    assert np.all(overlap <= hi + 1e-9)


def test_stitch_identical_images_has_no_seam():
    rng = np.random.default_rng(6)
    img = rng.random((20, 25)) * 255.0
    pano = stitch_images(img, img, shift=(0, 0))
    assert pano.shape == (20, 25)
    assert np.allclose(pano, img)


def test_stitch_no_overlap_still_valid_canvas():
    rng = np.random.default_rng(7)
    img1 = rng.random((10, 12)) * 255.0
    img2 = rng.random((10, 12)) * 255.0
    pano = stitch_images(img1, img2, shift=(0, 20))
    assert pano.shape == (10, 32)
    assert np.array_equal(pano[:, 0:12], img1)
    assert np.array_equal(pano[:, 20:32], img2)
    assert np.all(pano[:, 12:20] == 0.0)


def test_colour_images_shift_and_stitch():
    rng = np.random.default_rng(8)
    dy, dx = 3, 6
    base = rng.random((23, 26, 3)) * 255.0
    img1 = base[0:20, 0:20, :].copy()
    img2 = base[3:23, 6:26, :].copy()
    found, score = find_best_shift(img1, img2, max_shift=10)
    assert found == (3, 6)
    assert score == pytest.approx(1.0)
    pano = stitch_images(img1, img2, shift=found)
    assert pano.shape == (23, 26, 3)
    assert np.array_equal(pano[0:3, 0:20, :], img1[0:3, :, :])


def test_invalid_1d_image_raises():
    with pytest.raises(ValueError):
        find_best_shift(np.zeros(10), np.zeros((10, 10)))


def test_invalid_empty_image_raises():
    with pytest.raises(ValueError):
        stitch_images(np.zeros((0, 5)), np.zeros((5, 5)))


def test_invalid_mismatched_channels_raises():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        stitch_images(rng.random((10, 10, 3)), rng.random((10, 10, 4)))


def test_invalid_nan_raises():
    img = np.zeros((10, 10))
    img[3, 3] = np.nan
    with pytest.raises(ValueError):
        find_best_shift(img, np.zeros((10, 10)))


def test_invalid_negative_max_shift_raises():
    img1, img2 = make_pair()
    with pytest.raises(ValueError):
        find_best_shift(img1, img2, max_shift=-1)


def test_invalid_shift_type_raises():
    img1, img2 = make_pair()
    with pytest.raises(ValueError):
        stitch_images(img1, img2, shift=(1.5, 0))
    with pytest.raises(ValueError):
        stitch_images(img1, img2, shift=(1, 2, 3))


def test_determinism():
    img1, img2 = make_pair(shift=(4, 7))
    first = find_best_shift(img1, img2, max_shift=12)
    second = find_best_shift(img1, img2, max_shift=12)
    assert first == second
    pano1 = stitch_images(img1, img2, shift=(4, 7))
    pano2 = stitch_images(img1, img2, shift=(4, 7))
    assert np.array_equal(pano1, pano2)
