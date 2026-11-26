import math

import numpy as np
import pytest

from image_watermark.model import (
    add_visible_watermark,
    embed_invisible_watermark,
    psnr,
    verify_invisible_watermark,
)
from image_watermark.demo import make_test_image


def _image(height=64, width=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3)).astype(np.uint8)


def test_visible_watermark_changes_only_chosen_corner():
    img = _image()
    out = add_visible_watermark(img, alpha=0.3, position="bottom-right",
                                block_size=8)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Corner region is 4 * block_size = 32 px square, bottom-right.
    assert np.array_equal(out[:32], img[:32])
    assert np.array_equal(out[32:, :32], img[32:, :32])
    assert not np.array_equal(out[32:, 32:], img[32:, 32:])
    # Input is not modified.
    assert np.array_equal(img, _image())


def test_visible_watermark_other_positions():
    img = _image()
    out = add_visible_watermark(img, alpha=0.5, position="top-left",
                                block_size=8)
    assert not np.array_equal(out[:32, :32], img[:32, :32])
    assert np.array_equal(out[32:], img[32:])
    assert np.array_equal(out[:, 32:], img[:, 32:])


def test_visible_watermark_grayscale():
    img = np.random.default_rng(1).integers(
        0, 256, size=(48, 48)).astype(np.uint8)
    out = add_visible_watermark(img, alpha=0.4, position="top-right")
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[32:, :16], img[32:, :16])  # outside corner
    assert not np.array_equal(out[:32, 16:], img[:32, 16:])


def test_alpha_strength_difference():
    img = _image()
    weak = add_visible_watermark(img, alpha=0.1, position="bottom-right")
    strong = add_visible_watermark(img, alpha=1.0, position="bottom-right")
    corner = (slice(32, None), slice(32, None))
    weak_change = np.abs(weak[corner].astype(float)
                         - img[corner].astype(float)).mean()
    strong_change = np.abs(strong[corner].astype(float)
                           - img[corner].astype(float)).mean()
    assert strong_change > weak_change
    assert strong_change > 50  # alpha=1 drives white squares to 255
    # With alpha=1, the white checkerboard squares are exactly 255.
    assert strong[32, 32, 0] == 255


def test_invalid_position_raises():
    with pytest.raises(ValueError):
        add_visible_watermark(_image(), position="centre")
    with pytest.raises(ValueError):
        add_visible_watermark(_image(), position="Bottom-Right")


def test_invalid_alpha_raises():
    for bad in (0, -0.1, 1.5, 2.0):
        with pytest.raises(ValueError):
            add_visible_watermark(_image(), alpha=bad)


def test_invisible_embed_and_verify_correct_key():
    img = _image()
    hidden = embed_invisible_watermark(img, "daily-builds")
    assert hidden.shape == img.shape and hidden.dtype == np.uint8
    assert verify_invisible_watermark(hidden, "daily-builds") is True


def test_invisible_verify_wrong_key_false():
    img = _image()
    hidden = embed_invisible_watermark(img, "daily-builds")
    assert verify_invisible_watermark(hidden, "wrong-key") is False
    assert verify_invisible_watermark(hidden, "daily-build") is False


def test_verify_false_on_unwatermarked_image():
    img = _image()
    assert verify_invisible_watermark(img, "daily-builds") is False


def test_invisible_watermark_barely_changes_image():
    img = _image()
    hidden = embed_invisible_watermark(img, "daily-builds")
    assert psnr(img, hidden) > 40
    assert np.abs(hidden.astype(int) - img.astype(int)).max() <= 1


def test_embed_too_small_image_raises():
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)  # 48 pixels < 272 bits
    with pytest.raises(ValueError):
        embed_invisible_watermark(tiny, "daily-builds")
    # Exactly 272 pixels holds one copy.
    just_right = np.zeros((16, 17), dtype=np.uint8)  # 272 pixels
    hidden = embed_invisible_watermark(just_right, "daily-builds")
    assert verify_invisible_watermark(hidden, "daily-builds") is True


def test_psnr_identical_images_is_inf():
    img = _image()
    assert psnr(img, img) == float("inf")
    assert math.isinf(psnr(img, img.copy()))


def test_psnr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        psnr(_image(), _image(height=32))
    with pytest.raises(ValueError):
        psnr(_image(), np.zeros((64, 64), dtype=np.uint8))


def test_empty_key_raises():
    img = _image()
    with pytest.raises(ValueError):
        embed_invisible_watermark(img, "")
    with pytest.raises(ValueError):
        verify_invisible_watermark(img, "")
    with pytest.raises(TypeError):
        embed_invisible_watermark(img, 123)


def test_non_uint8_input_raises():
    float_img = np.zeros((64, 64, 3), dtype=np.float64)
    with pytest.raises(TypeError):
        add_visible_watermark(float_img)
    with pytest.raises(TypeError):
        embed_invisible_watermark(float_img, "daily-builds")
    with pytest.raises(TypeError):
        verify_invisible_watermark(float_img, "daily-builds")
    with pytest.raises(TypeError):
        psnr(float_img, float_img)
    with pytest.raises(TypeError):
        add_visible_watermark([[1, 2], [3, 4]])  # not an ndarray


def test_demo_behaviour():
    img = make_test_image()
    hidden = embed_invisible_watermark(img, "daily-builds")
    assert verify_invisible_watermark(hidden, "daily-builds") is True
    assert verify_invisible_watermark(hidden, "wrong-key") is False
    assert psnr(img, hidden) > 40
    vis = add_visible_watermark(img)
    assert np.array_equal(vis[:32], img[:32])
