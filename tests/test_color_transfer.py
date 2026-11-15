"""Tests for color_transfer."""

import numpy as np
import pytest

from color_transfer import (
    color_transfer,
    lab_to_rgb,
    rgb_to_lab,
    transfer_rgb,
)
from color_transfer.demo import make_gradient


def _random_image(seed=0, shape=(12, 14, 3)):
    rng = np.random.default_rng(seed)
    return rng.random(shape) * 255.0


def test_transfer_onto_itself_is_identity():
    image = _random_image()
    out = color_transfer(image, image)
    assert np.allclose(out, image, atol=1.0)


def test_lab_result_means_match_target():
    source = _random_image(0)
    target = _random_image(1) * 0.5 + 60.0
    out = color_transfer(source, target, colorspace="lab")
    src_lab = rgb_to_lab(source)
    tgt_lab = rgb_to_lab(np.clip(target, 0, 255))
    out_lab = rgb_to_lab(out)
    assert np.allclose(out_lab.mean(axis=(0, 1)),
                       tgt_lab.mean(axis=(0, 1)), atol=2.0)


def test_lab_result_stds_match_target():
    source = _random_image(0)
    target = _random_image(1) * 0.5 + 60.0
    out = color_transfer(source, target, colorspace="lab")
    tgt_lab = rgb_to_lab(np.clip(target, 0, 255))
    out_lab = rgb_to_lab(out)
    assert np.allclose(out_lab.std(axis=(0, 1)),
                       tgt_lab.std(axis=(0, 1)), rtol=0.15)


def test_rgb_transfer_matches_means():
    source = _random_image(2)
    target = _random_image(3) * 0.6 + 40.0
    out = transfer_rgb(source, target)
    assert np.allclose(out.mean(axis=(0, 1)),
                       np.clip(target, 0, 255).mean(axis=(0, 1)), atol=2.0)
    assert np.allclose(out.std(axis=(0, 1)),
                       np.clip(target, 0, 255).std(axis=(0, 1)), rtol=0.1)


def test_rgb_lab_round_trip():
    image = _random_image(4)
    back = lab_to_rgb(rgb_to_lab(image))
    assert np.allclose(back, image, atol=1.0)


def test_output_clipped_to_valid_range():
    source = make_gradient(10, 12, base=(250.0, 10.0, 10.0),
                           slope=(0.1, 0.1, 0.1))
    target = make_gradient(10, 12, base=(10.0, 10.0, 250.0),
                           slope=(0.1, 0.1, 0.1))
    out = color_transfer(source, target)
    assert out.min() >= 0.0
    assert out.max() <= 255.0
    assert out.shape == source.shape


def test_different_shapes_allowed():
    source = _random_image(5, shape=(10, 12, 3))
    target = _random_image(6, shape=(8, 9, 3))
    out = color_transfer(source, target)
    assert out.shape == source.shape


def test_white_and_black_round_trip():
    image = np.array([[[255.0, 255.0, 255.0], [0.0, 0.0, 0.0]]])
    back = lab_to_rgb(rgb_to_lab(image))
    assert np.allclose(back, image, atol=1.0)


def test_constant_source_channel_raises():
    source = np.full((6, 6, 3), 128.0)
    target = _random_image(7, shape=(6, 6, 3))
    with pytest.raises(ValueError):
        color_transfer(source, target)


def test_2d_image_raises():
    with pytest.raises(ValueError):
        color_transfer(np.zeros((5, 5)), _random_image(8, (5, 5, 3)))


def test_four_channel_target_raises():
    with pytest.raises(ValueError):
        color_transfer(_random_image(9), np.zeros((5, 5, 4)))


def test_bad_colorspace_raises():
    with pytest.raises(ValueError):
        color_transfer(_random_image(10), _random_image(11),
                       colorspace="hsv")


def test_out_of_range_values_raise():
    source = _random_image(12)
    source[0, 0, 0] = 300.0
    with pytest.raises(ValueError):
        color_transfer(source, _random_image(13))
    with pytest.raises(ValueError):
        color_transfer(_random_image(14), _random_image(15) - 300.0)


def test_empty_image_raises():
    with pytest.raises(ValueError):
        color_transfer(np.zeros((0, 0, 3)), _random_image(16, (4, 4, 3)))
