"""Tests for stereo_disparity."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stereo_disparity import (
    block_match,
    depth_order_ok,
    disparity_accuracy,
    make_synthetic_pair,
)


def test_synthetic_pair_shapes_and_truth():
    left, right, truth = make_synthetic_pair()
    assert left.shape == right.shape == truth.shape == (60, 80)
    assert truth[15, 15] == 3
    assert truth[45, 55] == 8
    assert truth[0, 0] == 0
    assert left[15, 15] == pytest.approx(200.0, abs=6.0)
    assert left[45, 55] == pytest.approx(60.0, abs=6.0)


def test_block_match_recovers_known_disparities():
    left, right, truth = make_synthetic_pair()
    disp = block_match(left, right, max_disparity=12, window_radius=2)
    assert disp.shape == (60, 80)
    assert disp.dtype == np.int32
    near = disp[37:53, 47:63]
    far = disp[12:28, 12:28]
    assert np.median(near[near >= 0]) == 8
    assert np.median(far[far >= 0]) == 3
    assert disparity_accuracy(disp, truth) > 0.8


def test_identical_images_give_zero_disparity():
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 20.0, size=(30, 40))
    disp = block_match(img, img, max_disparity=8, window_radius=2)
    valid = disp[disp >= 0]
    assert np.mean(valid == 0) > 0.95


def test_border_pixels_are_unknown():
    left, right, _truth = make_synthetic_pair()
    disp = block_match(left, right, max_disparity=12, window_radius=2)
    assert np.all(disp[:2, :] == -1)
    assert np.all(disp[-2:, :] == -1)
    assert np.all(disp[:, :14] == -1)
    assert np.all(disp[:, -2:] == -1)


def test_depth_order_ok():
    left, right, _truth = make_synthetic_pair()
    disp = block_match(left, right, max_disparity=12, window_radius=2)
    assert depth_order_ok(disp, (slice(37, 53), slice(47, 63)),
                          (slice(12, 28), slice(12, 28)))
    assert not depth_order_ok(disp, (slice(12, 28), slice(12, 28)),
                              (slice(37, 53), slice(47, 63)))


def test_depth_order_ok_empty_region_raises():
    disp = np.full((10, 10), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="no valid disparities"):
        depth_order_ok(disp, (slice(0, 5), slice(0, 5)),
                       (slice(5, 10), slice(5, 10)))


def test_disparity_accuracy_known():
    disp = np.array([[0, 3, -1], [8, 8, 3]])
    truth = np.array([[0, 3, 5], [8, 7, 3]])
    assert disparity_accuracy(disp, truth) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="same shape"):
        disparity_accuracy(disp, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="no valid disparities"):
        disparity_accuracy(np.full((2, 2), -1), np.zeros((2, 2), dtype=int))


def test_max_disparity_zero():
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 20.0, size=(20, 30))
    disp = block_match(img, img, max_disparity=0, window_radius=1)
    valid = disp[disp >= 0]
    assert np.all(valid == 0)


def test_validation_errors():
    left, right, _truth = make_synthetic_pair()
    with pytest.raises(ValueError, match="same shape"):
        block_match(left, np.zeros((10, 10)))
    with pytest.raises(ValueError, match="2D"):
        block_match(np.zeros((5, 5, 3)), right)
    with pytest.raises(ValueError, match="non-empty"):
        block_match(np.zeros((0, 0)), np.zeros((0, 0)))
    with pytest.raises(ValueError, match="max_disparity"):
        block_match(left, right, max_disparity=-1)
    with pytest.raises(ValueError, match="window_radius"):
        block_match(left, right, window_radius=-1)
    with pytest.raises(ValueError, match="too narrow"):
        block_match(np.zeros((10, 12)), np.zeros((10, 12)),
                    max_disparity=10, window_radius=2)
