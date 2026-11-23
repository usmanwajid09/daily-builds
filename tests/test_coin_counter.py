"""Tests for coin_counter (distance-transform coin counting, numpy only)."""

import numpy as np
import pytest

from coin_counter.coins import (
    RADIUS_TO_VALUE,
    VALUE_TO_RADIUS,
    CoinInfo,
    count_coins,
    distance_transform,
    make_coins_image,
)


def _scene(coins, size=200, seed=7, **kwargs):
    return make_coins_image(coins, size=size, seed=seed, **kwargs)


# --- make_coins_image ---

def test_generated_image_shape_and_dtype():
    img = _scene([(50, 50, 10)])
    assert img.shape == (200, 200)
    assert img.dtype == np.uint8


def test_generated_image_has_bright_coin_on_dark_background():
    img = _scene([(100, 100, 14)], noise_level=0)
    assert img[100, 100] == 220  # coin centre is bright
    assert img[0, 0] == 20       # far corner is background


def test_make_coins_image_by_denomination():
    img = _scene([(50, 50, 5), (150, 150, 10)], by="denomination")
    coins, summary = count_coins(img)
    assert summary["count"] == 2
    assert summary["total_value"] == 15
    assert summary["by_denomination"] == {5: 1, 10: 1}


def test_make_coins_image_same_seed_is_reproducible():
    a = _scene([(50, 50, 10), (150, 120, 18)], seed=3)
    b = _scene([(50, 50, 10), (150, 120, 18)], seed=3)
    assert np.array_equal(a, b)


def test_make_coins_image_rejects_unknown_denomination():
    with pytest.raises(ValueError, match="unknown denomination"):
        _scene([(50, 50, 2)], by="denomination")


def test_make_coins_image_rejects_coin_outside_image():
    with pytest.raises(ValueError, match="does not fit"):
        _scene([(195, 100, 10)])  # radius 10 coin pokes past the edge
    with pytest.raises(ValueError, match="does not fit"):
        _scene([(100, 100, 250)])


def test_make_coins_image_rejects_bad_radius_and_size():
    with pytest.raises(ValueError, match="radius must be >= 1"):
        _scene([(50, 50, 0)])
    with pytest.raises(ValueError, match="size"):
        make_coins_image([(50, 50, 10)], size=10)
    with pytest.raises(ValueError, match="by must be"):
        _scene([(50, 50, 10)], by="diameter")


# --- distance_transform ---

def test_distance_transform_single_disc_peak_is_radius():
    binary = np.zeros((101, 101), dtype=np.uint8)
    rows, cols = np.indices(binary.shape)
    binary[(rows - 50) ** 2 + (cols - 50) ** 2 <= 14 ** 2] = 1
    dist = distance_transform(binary)
    assert dist.shape == binary.shape
    assert dist[50, 50] == pytest.approx(14.0, abs=1.5)
    assert dist[0, 0] == 0.0  # background is zero


def test_distance_transform_rejects_all_foreground():
    with pytest.raises(ValueError, match="no background"):
        distance_transform(np.ones((5, 5), dtype=np.uint8))


def test_distance_transform_rejects_non_binary():
    with pytest.raises(ValueError, match="only contain 0 and 1"):
        distance_transform(np.array([[0, 2], [1, 0]]))


# --- count_coins: core behaviour ---

def test_empty_image_has_zero_coins():
    coins, summary = count_coins(np.zeros((100, 100), dtype=np.uint8))
    assert coins == []
    assert summary == {"count": 0, "total_value": 0, "by_denomination": {}}


def test_single_coin_detected_with_correct_radius_and_value():
    coins, summary = count_coins(_scene([(100, 100, 14)]))
    assert len(coins) == 1
    coin = coins[0]
    assert (coin["x"], coin["y"]) == (100, 100)
    assert coin["radius"] == pytest.approx(14.0, abs=1.5)
    assert coin["denomination"] == 5
    assert summary["total_value"] == 5


def test_each_denomination_detected():
    for radius, value in RADIUS_TO_VALUE.items():
        coins, summary = count_coins(_scene([(100, 100, radius)]))
        assert len(coins) == 1
        assert coins[0]["denomination"] == value
        assert summary["total_value"] == value


def test_multiple_coins_exact_count_and_total():
    scene = [(40, 40, 10), (100, 35, 10), (160, 45, 10),
             (50, 130, 14), (130, 120, 14), (165, 155, 18)]
    coins, summary = count_coins(_scene(scene))
    assert summary["count"] == 6
    assert summary["total_value"] == 3 * 1 + 2 * 5 + 1 * 10
    assert summary["by_denomination"] == {1: 3, 5: 2, 10: 1}
    assert len(coins) == 6


def test_coin_info_supports_attribute_access():
    coins, _ = count_coins(_scene([(60, 80, 10)]))
    assert isinstance(coins[0], CoinInfo)
    assert coins[0].x == 60 and coins[0].y == 80
    assert coins[0].denomination == 1


def test_rgb_image_counts_the_same_as_grayscale():
    gray = _scene([(40, 40, 10), (150, 150, 18)])
    rgb = np.stack([gray, gray, gray], axis=2)
    coins_rgb, summary_rgb = count_coins(rgb)
    _, summary_gray = count_coins(gray)
    assert summary_rgb == summary_gray
    assert len(coins_rgb) == 2


def test_normalised_float_image_counts():
    gray = _scene([(50, 50, 10), (150, 100, 14)])
    coins, summary = count_coins(gray.astype(np.float64) / 255.0,
                                 threshold=0.5)
    assert summary["count"] == 2
    assert summary["total_value"] == 6


def test_slightly_overlapping_coins_still_both_found():
    # Centres 24 px apart; radii 14 + 14 = 28, so the discs overlap
    # by a few pixels but their distance-transform peaks stay apart.
    coins, summary = count_coins(
        _scene([(80, 100, 14), (104, 100, 14)], noise_level=0)
    )
    assert summary["count"] == 2
    assert summary["total_value"] == 10


def test_noisy_scene_still_exact():
    scene = [(40, 40, 10), (150, 50, 18), (100, 150, 14)]
    for seed in (1, 2, 3):
        _, summary = count_coins(_scene(scene, seed=seed))
        assert summary["count"] == 3
        assert summary["total_value"] == 16


def test_unknown_radius_coin_counts_but_has_no_value():
    # Radius 22 is not in the denomination map: counted, value 0.
    coins, summary = count_coins(_scene([(100, 100, 22)]))
    assert summary["count"] == 1
    assert coins[0]["denomination"] is None
    assert summary["total_value"] == 0
    assert summary["by_denomination"] == {}


def test_tiny_speckle_noise_is_ignored():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10, 10] = 220  # single bright pixel, radius ~1
    img[50, 60:62] = 220
    coins, summary = count_coins(img)
    assert summary["count"] == 0


# --- count_coins: error paths ---

def test_count_coins_rejects_bad_image_shapes():
    with pytest.raises(ValueError, match="must not be empty"):
        count_coins(np.zeros((0, 0), dtype=np.uint8))
    with pytest.raises(ValueError, match="2D grayscale or 3D RGB"):
        count_coins(np.zeros((5, 5, 5, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="must have 3"):
        count_coins(np.zeros((5, 5, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="must be a numpy array"):
        count_coins([[0, 1], [1, 0]])


def test_count_coins_rejects_bad_threshold():
    img = _scene([(50, 50, 10)])
    with pytest.raises(ValueError, match="threshold must be in"):
        count_coins(img, threshold=0)
    with pytest.raises(ValueError, match="threshold must be in"):
        count_coins(img, threshold=300)
    with pytest.raises(ValueError, match="threshold must be in"):
        count_coins(img.astype(np.float64) / 255.0, threshold=2.0)


def test_count_coins_rejects_bad_min_radius():
    img = _scene([(50, 50, 10)])
    with pytest.raises(ValueError, match="min_radius must be >= 1"):
        count_coins(img, min_radius=0)
    with pytest.raises(ValueError, match="min_radius must be an integer"):
        count_coins(img, min_radius=2.5)


def test_count_coins_rejects_all_foreground_image():
    with pytest.raises(ValueError, match="no background"):
        count_coins(np.full((20, 20), 255, dtype=np.uint8))


def test_value_to_radius_is_inverse_of_radius_to_value():
    for radius, value in RADIUS_TO_VALUE.items():
        assert VALUE_TO_RADIUS[value] == radius
