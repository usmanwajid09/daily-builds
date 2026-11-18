"""Tests for orb_features."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orb_features import (
    corner_score,
    describe,
    detect_keypoints,
    extract_orb,
    hamming_distance,
    is_corner,
    make_descriptor_pairs,
    match_descriptors,
)
from orb_features.demo import make_image, shift_image


def square_image():
    img = np.zeros((60, 60), dtype=np.float64)
    img[20:40, 20:40] = 255.0
    return img


def test_square_corners_detected():
    img = square_image()
    corners = detect_keypoints(img, threshold=20, n_contiguous=9)
    assert len(corners) >= 4
    for r, c in [(20, 20), (20, 39), (39, 20), (39, 39)]:
        assert any(abs(kr - r) <= 2 and abs(kc - c) <= 2 for kr, kc in corners)


def test_uniform_image_has_no_corners():
    img = np.full((40, 40), 128.0)
    assert detect_keypoints(img) == []


def test_is_corner_and_score():
    img = square_image()
    assert is_corner(img, 20, 20, threshold=20, n_contiguous=9)
    assert not is_corner(img, 30, 30, threshold=20, n_contiguous=9)
    assert corner_score(img, 20, 20, threshold=20, n_contiguous=9) > 0
    assert corner_score(img, 30, 30) == 0.0
    assert corner_score(img, 0, 0) == 0.0  # border pixel: no circle
    assert not is_corner(img, 0, 0)


def test_detect_keypoints_sorted_strongest_first():
    img = make_image()
    corners = detect_keypoints(img, threshold=20, n_contiguous=9)
    scores = [corner_score(img, r, c, threshold=20, n_contiguous=9)
              for r, c in corners]
    assert scores == sorted(scores, reverse=True)


def test_descriptor_pairs_deterministic_and_bounded():
    p1 = make_descriptor_pairs(64, patch_radius=15, seed=3)
    p2 = make_descriptor_pairs(64, patch_radius=15, seed=3)
    assert np.array_equal(p1, p2)
    assert p1.shape == (64, 2, 2)
    assert np.abs(p1).max() <= 15
    with pytest.raises(ValueError, match="n_bits"):
        make_descriptor_pairs(0)
    with pytest.raises(ValueError, match="patch_radius"):
        make_descriptor_pairs(16, patch_radius=0)


def test_describe_shape_and_determinism():
    img = make_image()
    kps = detect_keypoints(img, threshold=20, n_contiguous=9)
    kept, desc = describe(img, kps)
    assert len(kept) == desc.shape[0]
    assert desc.shape[1] == 256
    assert desc.dtype == bool
    kept2, desc2 = describe(img, kps)
    assert np.array_equal(desc, desc2)


def test_describe_drops_border_keypoints():
    img = make_image()
    kept, desc = describe(img, [(2, 2), (30, 30)])
    assert (2, 2) not in kept
    assert (30, 30) in kept
    assert desc.shape[0] == 1


def test_hamming_distance_known():
    a = np.array([[True, False, True], [False, False, True]])
    b = np.array([[True, False, False], [True, True, True]])
    d = hamming_distance(a, b)
    assert d.tolist() == [[1, 1], [2, 2]]
    with pytest.raises(ValueError, match="lengths differ"):
        hamming_distance(a, np.zeros((1, 2), dtype=bool))
    with pytest.raises(ValueError, match="empty"):
        hamming_distance(np.zeros((0, 3), dtype=bool), b)


def test_shifted_image_matches_recover_shift():
    img1 = make_image()
    img2 = shift_image(img1, 5, 8)
    kp1, desc1 = extract_orb(img1, n_contiguous=9)
    kp2, desc2 = extract_orb(img2, n_contiguous=9)
    assert len(kp1) > 0 and len(kp2) > 0
    matches = match_descriptors(desc1, desc2)
    assert len(matches) >= 4
    shifts = np.array([(kp2[j][0] - kp1[i][0], kp2[j][1] - kp1[i][1])
                       for i, j, _d in matches])
    assert np.median(shifts, axis=0).tolist() == [5, 8]


def test_ratio_test_filters_ambiguous_matches():
    # Two identical descriptors in desc2 make the best match ambiguous.
    desc1 = np.zeros((1, 8), dtype=bool)
    desc1[0, :4] = True
    desc2 = np.zeros((2, 8), dtype=bool)
    desc2[:, :4] = True
    assert match_descriptors(desc1, desc2, ratio=0.75) == []
    assert len(match_descriptors(desc1, desc2, ratio=None)) == 1


def test_validation_errors():
    with pytest.raises(ValueError, match="2D"):
        detect_keypoints(np.zeros((5, 5, 3)))
    with pytest.raises(ValueError, match="non-empty"):
        detect_keypoints(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="threshold"):
        detect_keypoints(np.zeros((10, 10)), threshold=0)
    with pytest.raises(ValueError, match="n_contiguous"):
        detect_keypoints(np.zeros((10, 10)), n_contiguous=17)
    with pytest.raises(ValueError, match="max_keypoints"):
        detect_keypoints(np.zeros((10, 10)), max_keypoints=0)
    with pytest.raises(ValueError, match="outside the image"):
        is_corner(np.zeros((10, 10)), 20, 5)


def test_extract_orb_empty_image_returns_empty():
    kept, desc = extract_orb(np.full((40, 40), 128.0))
    assert kept == []
    assert desc.shape == (0, 256)
