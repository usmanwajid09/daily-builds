"""Tests for mean_shift_segmentation."""
import numpy as np
import pytest

from mean_shift_segmentation import (
    mean_shift_segment,
    segment_count,
    segment_image,
)
from mean_shift_segmentation.demo import build_scene


def two_block_image():
    """Left half red, right half blue — two clear segments."""
    img = np.zeros((16, 24, 3), dtype=np.uint8)
    img[:, :12] = (220, 30, 30)
    img[:, 12:] = (30, 60, 220)
    return img


def test_two_blocks_two_segments():
    labels = mean_shift_segment(two_block_image(), bandwidth=0.25)
    assert labels.shape == (16, 24)
    assert segment_count(labels) == 2
    # Each half is internally uniform.
    assert len(np.unique(labels[:, :12])) == 1
    assert len(np.unique(labels[:, 12:])) == 1
    assert labels[0, 0] != labels[0, -1]


def test_flat_image_one_segment():
    img = np.full((12, 12, 3), 128, dtype=np.uint8)
    labels = mean_shift_segment(img)
    assert segment_count(labels) == 1
    assert (labels == 0).all()


def test_grayscale_input_works():
    gray = np.zeros((12, 16), dtype=np.uint8)
    gray[:, 8:] = 200
    labels = mean_shift_segment(gray, bandwidth=0.25)
    assert labels.shape == (12, 16)
    assert segment_count(labels) == 2


def test_labels_are_int_and_cover_all():
    labels = mean_shift_segment(two_block_image())
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == set(range(segment_count(labels)))


def test_larger_bandwidth_fewer_segments():
    img = build_scene()
    few = segment_count(mean_shift_segment(img, bandwidth=0.6))
    many = segment_count(mean_shift_segment(img, bandwidth=0.15))
    assert few <= many


def test_segment_image_shape_dtype_and_few_colours():
    img = build_scene()
    out = segment_image(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == segment_count(
        mean_shift_segment(img)
    )


def test_segment_image_means_match_segments():
    img = two_block_image()
    out = segment_image(img, bandwidth=0.25)
    assert np.allclose(out[:, :12], (220, 30, 30), atol=1)
    assert np.allclose(out[:, 12:], (30, 60, 220), atol=1)


def test_spatial_weight_zero_still_segments_by_colour():
    labels = mean_shift_segment(two_block_image(), spatial_weight=0.0)
    assert segment_count(labels) == 2


def test_invalid_bandwidth_raises():
    with pytest.raises(ValueError):
        mean_shift_segment(two_block_image(), bandwidth=0)
    with pytest.raises(ValueError):
        mean_shift_segment(two_block_image(), bandwidth=-1)


def test_invalid_spatial_weight_raises():
    with pytest.raises(ValueError):
        mean_shift_segment(two_block_image(), spatial_weight=-0.5)


def test_invalid_max_iter_and_tol_raise():
    with pytest.raises(ValueError):
        mean_shift_segment(two_block_image(), max_iter=0)
    with pytest.raises(ValueError):
        mean_shift_segment(two_block_image(), tol=0)


def test_non_array_and_empty_raise():
    with pytest.raises(TypeError):
        mean_shift_segment([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        mean_shift_segment(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        mean_shift_segment(np.zeros((4, 4, 4), dtype=np.uint8))


def test_segment_count_empty_raises():
    with pytest.raises(ValueError):
        segment_count(np.array([], dtype=np.int32))


def test_demo_scene_segments():
    scene = build_scene()
    labels = mean_shift_segment(scene, bandwidth=0.25, spatial_weight=0.5)
    assert 3 <= segment_count(labels) <= 6
