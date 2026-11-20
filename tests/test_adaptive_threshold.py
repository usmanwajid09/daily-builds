"""Tests for adaptive_threshold."""

import numpy as np
import pytest

from adaptive_threshold import (
    adaptive_threshold,
    compare_thresholds,
    global_threshold,
)
from adaptive_threshold.demo import bars_mask, make_test_image


def test_output_shape_and_dtype():
    image = make_test_image()
    mask = adaptive_threshold(image)
    assert mask.shape == image.shape
    assert mask.dtype == bool


def test_constant_image_with_zero_c():
    image = np.full((12, 12), 100.0)
    assert not adaptive_threshold(image, c=0.0).any()
    # With C > 0 the flat region is above (local_mean - C).
    assert adaptive_threshold(image, c=2.0).all()


def test_bright_spot_detected():
    image = np.full((21, 21), 50.0)
    image[9:12, 9:12] = 200.0
    mask = adaptive_threshold(image, block_size=11, c=0.0)
    assert mask[10, 10]
    assert not mask[0, 0]
    assert not mask[20, 20]


def test_gaussian_method_works():
    image = make_test_image()
    mask = adaptive_threshold(image, block_size=15, c=5.0,
                              method="gaussian")
    assert mask.shape == image.shape
    assert mask.dtype == bool
    # Both methods should agree on most pixels for this image.
    mean_mask = adaptive_threshold(image, block_size=15, c=5.0,
                                   method="mean")
    assert (mask != mean_mask).mean() < 0.2


def test_finds_bars_and_keeps_background():
    image = make_test_image()
    bars = bars_mask()
    mask = adaptive_threshold(image, block_size=15, c=10.0)
    assert mask[bars].mean() < 0.1   # bars are ink (False)
    assert mask[~bars].mean() > 0.95  # background is paper (True)


def test_global_threshold_fails_on_dark_side():
    # The global threshold calls the whole dark left background
    # "ink", while the adaptive one keeps it as paper.
    image = make_test_image()
    bars = bars_mask()
    glob = global_threshold(image)
    adapt = adaptive_threshold(image, block_size=15, c=10.0)
    assert glob[~bars].mean() < 0.9
    assert adapt[~bars].mean() > 0.95
    assert glob[bars].mean() < 0.1
    assert adapt[bars].mean() < 0.1


def test_invalid_block_size():
    image = make_test_image()
    with pytest.raises(ValueError):
        adaptive_threshold(image, block_size=10)
    with pytest.raises(ValueError):
        adaptive_threshold(image, block_size=1)
    with pytest.raises(ValueError):
        adaptive_threshold(image, block_size=0)


def test_invalid_c_and_method():
    image = make_test_image()
    with pytest.raises(ValueError):
        adaptive_threshold(image, c=-1.0)
    with pytest.raises(ValueError):
        adaptive_threshold(image, method="otsu")


def test_rejects_non_grayscale():
    with pytest.raises(ValueError):
        adaptive_threshold(np.zeros((10, 10, 3)))
    with pytest.raises(ValueError):
        adaptive_threshold(np.zeros((0, 0)))


def test_global_threshold_methods():
    image = np.array([[0.0, 100.0], [200.0, 255.0]])
    mean_mask = global_threshold(image, method="mean")
    assert mean_mask.tolist() == [[False, False], [True, True]]
    median_mask = global_threshold(image, method="median")
    assert median_mask.tolist() == [[False, False], [True, True]]
    with pytest.raises(ValueError):
        global_threshold(image, method="otsu")


def test_compare_thresholds_dict():
    image = make_test_image()
    result = compare_thresholds(image, block_size=15, c=5.0)
    assert set(result) == {
        "adaptive", "global", "adaptive_foreground",
        "global_foreground", "difference",
    }
    assert result["adaptive_foreground"] == int(result["adaptive"].sum())
    assert result["global_foreground"] == int(result["global"].sum())
    assert result["difference"] == int(
        (result["adaptive"] != result["global"]).sum())
    assert result["difference"] > 0  # the methods really differ here


def test_uint8_input_accepted():
    image = np.clip(make_test_image(), 0, 255).astype(np.uint8)
    mask = adaptive_threshold(image, block_size=11, c=2.0)
    assert mask.dtype == bool
    assert mask.shape == image.shape
