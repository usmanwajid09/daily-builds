"""Tests for dark_channel_dehazing."""
import numpy as np
import pytest

from dark_channel_dehazing.dehazing import (
    dark_channel,
    dehaze,
    estimate_atmospheric_light,
    estimate_transmission,
    make_clean_scene,
    make_hazy_image,
    recover,
)


def _scene_pair(seed=0):
    clean = make_clean_scene(size=64, seed=seed)
    hazy = make_hazy_image(clean, transmission=0.55,
                           atmospheric_light=(0.9, 0.9, 0.9))
    return clean, hazy


def test_dark_channel_white_image_is_one():
    dark = dark_channel(np.ones((16, 16, 3)))
    assert dark.shape == (16, 16)
    assert np.allclose(dark, 1.0)


def test_dark_channel_black_image_is_zero():
    assert np.allclose(dark_channel(np.zeros((16, 16, 3))), 0.0)


def test_dark_channel_uses_minimum_channel():
    img = np.zeros((16, 16, 3))
    img[..., 0] = 0.8  # red only: min channel is 0
    assert np.allclose(dark_channel(img, patch_size=3), 0.0)


def test_dark_channel_patch_spreads_dark_pixels():
    img = np.ones((16, 16, 3))
    img[8, 8] = 0.0  # one black pixel
    dark = dark_channel(img, patch_size=3)
    assert dark[8, 8] == 0.0
    assert dark[7, 7] == 0.0  # within the 3x3 patch
    assert dark[0, 0] == 1.0  # far away, unaffected


def test_dark_channel_validates_patch_size():
    with pytest.raises(ValueError):
        dark_channel(np.zeros((8, 8, 3)), patch_size=4)  # even
    with pytest.raises(ValueError):
        dark_channel(np.zeros((8, 8, 3)), patch_size=0)


def test_atmospheric_light_near_true_value():
    _, hazy = _scene_pair()
    A = estimate_atmospheric_light(hazy)
    assert A.shape == (3,)
    # True A is 0.9 in every channel. The scene has no near-white pixels,
    # so the classic estimator lands below 0.9 but must stay clearly high.
    assert np.all(A > 0.7)
    assert np.all(A <= 1.0)


def test_atmospheric_light_rejects_bad_fraction():
    _, hazy = _scene_pair()
    with pytest.raises(ValueError):
        estimate_atmospheric_light(hazy, top_fraction=0)


def test_transmission_range_and_clear_image():
    clean, hazy = _scene_pair()
    t = estimate_transmission(hazy, np.array([0.9, 0.9, 0.9]))
    assert t.shape == (64, 64)
    assert t.min() > 0 and t.max() <= 1.0
    # A haze-free image has a dark channel near 0, so t should be near 1.
    t_clear = estimate_transmission(clean, np.array([0.9, 0.9, 0.9]))
    assert t_clear.mean() > 0.9


def test_transmission_validates_omega():
    _, hazy = _scene_pair()
    with pytest.raises(ValueError):
        estimate_transmission(hazy, np.array([0.9, 0.9, 0.9]), omega=0)
    with pytest.raises(ValueError):
        estimate_transmission(hazy, np.array([0.9, 0.9, 0.9]), omega=1.5)


def test_recover_inverts_haze_model():
    clean, hazy = _scene_pair()
    recovered = recover(hazy, np.array([0.9, 0.9, 0.9]),
                        np.full((64, 64), 0.55))
    assert recovered.shape == clean.shape
    assert np.all(recovered >= 0) and np.all(recovered <= 1)
    assert np.mean((recovered - clean) ** 2) < 0.01


def test_recover_validates_inputs():
    _, hazy = _scene_pair()
    with pytest.raises(ValueError):
        recover(hazy, np.array([0.9, 0.9, 0.9]), np.full((16, 16), 0.5))
    with pytest.raises(ValueError):
        recover(hazy, np.array([0.9, 0.9, 0.9]), np.full((64, 64), 0.5), t0=0)
    with pytest.raises(ValueError):
        recover(hazy, np.array([0.9, 0.9]), np.full((64, 64), 0.5))


def test_dehaze_improves_mse():
    clean, hazy = _scene_pair()
    recovered, info = dehaze(hazy)
    assert recovered.shape == clean.shape
    assert np.all(recovered >= 0) and np.all(recovered <= 1)
    mse_hazy = np.mean((hazy - clean) ** 2)
    mse_rec = np.mean((recovered - clean) ** 2)
    assert mse_rec < mse_hazy
    assert abs(info["mean_transmission"] - 0.55) < 0.15
    assert np.all(info["atmospheric_light"] > 0.7)


def test_dehaze_clear_image_stays_nearly_unchanged():
    clean, _ = _scene_pair()
    recovered, _ = dehaze(clean)
    assert np.mean((recovered - clean) ** 2) < 0.01


def test_dehaze_accepts_uint8():
    clean, hazy = _scene_pair()
    hazy_u8 = (hazy * 255).astype(np.uint8)
    recovered, _ = dehaze(hazy_u8)
    assert recovered.shape == clean.shape
    assert recovered.dtype == np.float64


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        dark_channel(np.zeros((16, 16)))  # grayscale
    with pytest.raises(ValueError):
        dark_channel(np.zeros((0, 8, 3)))  # empty
    with pytest.raises(ValueError):
        dark_channel(np.full((8, 8, 3), 2.0))  # float out of range
    with pytest.raises(ValueError):
        make_hazy_image(np.zeros((8, 8, 3)), transmission=0)
    with pytest.raises(ValueError):
        make_hazy_image(np.zeros((8, 8, 3)), transmission=1.5)


def test_make_clean_scene_shape_and_range():
    scene = make_clean_scene(size=32, seed=1)
    assert scene.shape == (32, 32, 3)
    assert scene.min() >= 0.0 and scene.max() <= 1.0
    # The dark circle guarantees some near-zero dark-channel values.
    assert dark_channel(scene).min() < 0.1
