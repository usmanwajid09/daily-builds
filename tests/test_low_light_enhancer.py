"""Tests for low_light_enhancer."""
import numpy as np
import pytest

from low_light_enhancer.enhancer import (
    adaptive_gamma,
    clahe_enhance,
    denoise,
    enhance,
    gamma_correct,
    make_dark_image,
    metrics,
)


def test_metrics_keys_and_ranges():
    img = make_dark_image(size=32, seed=0)
    m = metrics(img)
    assert set(m) == {"brightness", "contrast", "dynamic_range", "min", "max"}
    assert 0.0 <= m["brightness"] <= 1.0
    assert m["contrast"] >= 0.0
    assert m["dynamic_range"] == pytest.approx(m["max"] - m["min"])
    assert m["brightness"] < 0.2  # the synthetic scene is dark


def test_metrics_grayscale():
    grey = np.full((16, 16), 0.5)
    m = metrics(grey)
    assert m["brightness"] == pytest.approx(0.5)
    assert m["contrast"] == pytest.approx(0.0)


def test_gamma_correct_brightens_with_gamma_below_one():
    dark = np.full((8, 8), 0.25)
    bright = gamma_correct(dark, 0.5)
    assert np.allclose(bright, 0.5)  # 0.25 ** 0.5 == 0.5
    assert bright.mean() > dark.mean()


def test_gamma_correct_identity_at_one():
    img = np.random.default_rng(0).random((8, 8, 3))
    assert np.allclose(gamma_correct(img, 1.0), img)


def test_gamma_correct_rejects_bad_gamma():
    with pytest.raises(ValueError):
        gamma_correct(np.zeros((4, 4)), 0)
    with pytest.raises(ValueError):
        gamma_correct(np.zeros((4, 4)), -1)


def test_adaptive_gamma_moves_mean_toward_target():
    dark = make_dark_image(size=32, seed=0)
    corrected, gamma = adaptive_gamma(dark, target_mean=0.5)
    assert gamma < 1.0  # dark image: must brighten
    assert metrics(corrected)["brightness"] > metrics(dark)["brightness"]
    assert metrics(corrected)["brightness"] > 0.3


def test_adaptive_gamma_bright_image_darkens():
    bright = np.full((16, 16, 3), 0.9)
    _, gamma = adaptive_gamma(bright, target_mean=0.5)
    assert gamma > 1.0


def test_adaptive_gamma_pure_black_unchanged():
    corrected, gamma = adaptive_gamma(np.zeros((8, 8, 3)))
    assert gamma == 1.0
    assert np.all(corrected == 0)


def test_adaptive_gamma_rejects_bad_target():
    with pytest.raises(ValueError):
        adaptive_gamma(np.zeros((4, 4)), target_mean=1.0)


def test_clahe_increases_contrast_of_flat_image():
    rng = np.random.default_rng(0)
    # Low-contrast image: all values squashed into [0.4, 0.5].
    img = 0.4 + 0.1 * rng.random((32, 32))
    before = metrics(img)["contrast"]
    after = metrics(clahe_enhance(img))["contrast"]
    assert after > before


def test_clahe_output_stays_in_range():
    img = np.random.default_rng(0).random((32, 32, 3))
    out = clahe_enhance(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_clahe_validates_params():
    with pytest.raises(ValueError):
        clahe_enhance(np.zeros((8, 8)), clip_limit=0)
    with pytest.raises(ValueError):
        clahe_enhance(np.zeros((8, 8)), tile_size=0)


def test_denoise_removes_salt_noise():
    rng = np.random.default_rng(0)
    flat = np.full((16, 16), 0.5)
    noisy = flat.copy()
    noisy[rng.random((16, 16)) < 0.1] = 0.0  # salt noise
    noisy[rng.random((16, 16)) < 0.1] = 1.0
    assert metrics(denoise(noisy))["contrast"] < metrics(noisy)["contrast"]


def test_denoise_preserves_flat_image():
    flat = np.full((16, 16, 3), 0.4)
    assert np.allclose(denoise(flat), flat)


def test_denoise_rejects_even_size():
    with pytest.raises(ValueError):
        denoise(np.zeros((8, 8)), size=4)


def test_enhance_improves_dark_image():
    dark = make_dark_image(size=64, seed=0)
    enhanced, before, after, gamma = enhance(dark)
    assert enhanced.shape == dark.shape
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0
    assert after["brightness"] > before["brightness"]
    assert after["contrast"] > before["contrast"]
    assert after["dynamic_range"] > before["dynamic_range"]
    assert gamma < 1.0


def test_enhance_works_on_grayscale():
    grey = make_dark_image(size=32, seed=0)[..., 0]
    enhanced, before, after, _ = enhance(grey)
    assert enhanced.shape == grey.shape
    assert after["brightness"] > before["brightness"]


def test_enhance_accepts_uint8():
    dark_u8 = (make_dark_image(size=32, seed=0) * 255).astype(np.uint8)
    enhanced, _, after, _ = enhance(dark_u8)
    assert enhanced.dtype == np.float64
    assert after["brightness"] > 0.2


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        metrics(np.zeros((8, 8, 4)))  # 4 channels
    with pytest.raises(ValueError):
        metrics(np.zeros((0, 8)))
    with pytest.raises(ValueError):
        metrics(np.full((8, 8), 2.0))  # float out of range
    with pytest.raises(ValueError):
        metrics(np.full((8, 8), np.nan))
    with pytest.raises(ValueError):
        make_dark_image(size=0)
