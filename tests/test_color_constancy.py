"""Tests for color_constancy."""

import numpy as np
import pytest

from color_constancy import (
    apply_gains,
    estimate_illuminant,
    gray_world_gains,
    shades_of_gray_gains,
    white_balance,
    white_patch_gains,
)
from color_constancy.demo import make_scene


def _neutral_image():
    rng = np.random.default_rng(1)
    base = rng.uniform(80, 180, size=(40, 50))
    return np.stack([base, base, base], axis=-1).astype(np.uint8)


def _cast_image():
    scene = make_scene()
    cast = np.array([1.45, 1.0, 0.55])
    return np.clip(scene.astype(np.float64) * cast, 0, 255).astype(np.uint8)


def test_gray_world_equalises_means():
    corrected = white_balance(_cast_image(), method="gray_world")
    means = corrected.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < 1.0


def test_gray_world_neutral_image_unchanged():
    img = _neutral_image()
    corrected = white_balance(img, method="gray_world")
    assert np.allclose(corrected, img, atol=2)


def test_gray_world_gains_make_means_equal():
    img = _cast_image()
    gains = gray_world_gains(img)
    assert gains.shape == (3,)
    assert np.all(gains > 0)
    # Red channel was boosted, so its gain must be smallest.
    assert gains[0] < gains[1] < gains[2]
    corrected = apply_gains(img, gains)
    means = corrected.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < 1.0


def test_white_patch_maps_peak_to_255():
    img = _cast_image()
    gains = white_patch_gains(img)
    corrected = apply_gains(img, gains)
    peaks = np.percentile(corrected.reshape(-1, 3), 99, axis=0)
    assert np.allclose(peaks, 255.0, atol=2.0)


def test_white_patch_percentile_100_uses_max():
    img = _cast_image()
    gains = white_patch_gains(img, percentile=100.0)
    corrected = apply_gains(img, gains)
    assert np.allclose(corrected.reshape(-1, 3).max(axis=0), 255.0, atol=1.0)


def test_shades_of_gray_reduces_cast():
    tinted = _cast_image()
    spread_before = tinted.reshape(-1, 3).mean(axis=0)
    spread_before = spread_before.max() - spread_before.min()
    corrected = white_balance(tinted, method="shades_of_gray", p=6)
    means = corrected.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < spread_before


def test_shades_of_gray_p1_matches_gray_world():
    img = _cast_image()
    assert np.allclose(
        shades_of_gray_gains(img, p=1), gray_world_gains(img), rtol=1e-6
    )


def test_correction_closer_to_original_than_tinted():
    scene = make_scene()
    tinted = _cast_image()
    # Gray-world and shades-of-gray recover the scene; white-patch
    # only anchors the brightest pixel, so it may over-brighten.
    for method in ("gray_world", "shades_of_gray"):
        corrected = white_balance(tinted, method=method)
        err_before = np.abs(tinted.astype(float) - scene).mean()
        err_after = np.abs(corrected.astype(float) - scene).mean()
        assert err_after < err_before


def test_estimate_illuminant_sums_to_one():
    illum = estimate_illuminant(_cast_image(), method="gray_world")
    assert illum.shape == (3,)
    assert illum.sum() == pytest.approx(1.0)
    assert np.all(illum > 0)
    # Warm cast: red is the strongest illuminant channel.
    assert illum[0] > illum[1] > illum[2]


def test_estimate_illuminant_neutral_is_one_third():
    illum = estimate_illuminant(_neutral_image())
    assert np.allclose(illum, 1.0 / 3.0, atol=0.02)


def test_apply_gains_scales_channels():
    img = np.full((4, 5, 3), 100, dtype=np.uint8)
    out = apply_gains(img, [2.0, 1.0, 0.5])
    assert np.all(out[:, :, 0] == 200)
    assert np.all(out[:, :, 1] == 100)
    assert np.all(out[:, :, 2] == 50)


def test_apply_gains_clips_at_255():
    img = np.full((2, 2, 3), 200, dtype=np.uint8)
    out = apply_gains(img, [2.0, 1.0, 1.0])
    assert np.all(out[:, :, 0] == 255)


def test_uint8_dtype_preserved():
    img = _cast_image()
    assert white_balance(img).dtype == np.uint8
    assert white_balance(img).shape == img.shape


def test_float01_round_trip():
    img = _cast_image().astype(np.float64) / 255.0
    out = white_balance(img, method="gray_world")
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_invalid_images_raise():
    gray = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        white_balance(gray)
    with pytest.raises(ValueError):
        white_balance(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        white_balance(np.zeros((10, 10, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        white_balance(np.zeros((10, 10, 3), dtype=np.float64) * 300.0)


def test_invalid_method_and_params_raise():
    img = _cast_image()
    with pytest.raises(ValueError):
        white_balance(img, method="von_kries")
    with pytest.raises(ValueError):
        estimate_illuminant(img, method="nope")
    with pytest.raises(ValueError):
        white_patch_gains(img, percentile=0)
    with pytest.raises(ValueError):
        white_patch_gains(img, percentile=101)
    with pytest.raises(ValueError):
        shades_of_gray_gains(img, p=0)
    with pytest.raises(ValueError):
        shades_of_gray_gains(img, p=-2)
    with pytest.raises(ValueError):
        apply_gains(img, [1.0, 1.0])
    with pytest.raises(ValueError):
        apply_gains(img, [1.0, -1.0, 1.0])


def test_black_channel_raises():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, 1:] = 100
    with pytest.raises(ValueError):
        gray_world_gains(img)
    with pytest.raises(ValueError):
        white_patch_gains(img)
