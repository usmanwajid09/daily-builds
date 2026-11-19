import numpy as np
import pytest

from chroma_key.chroma import (
    color_distance,
    composite,
    green_mask,
    replaced_pixel_count,
)
from chroma_key.demo import make_background, make_foreground


def test_pure_green_is_fully_replaced():
    fg = np.zeros((6, 8, 3))
    fg[:, :, 1] = 255.0
    bg = np.full((6, 8, 3), 42.0)
    assert np.allclose(composite(fg, bg), bg)


def test_non_green_pixels_are_kept():
    fg = np.full((6, 8, 3), 255.0)  # white, far from green
    bg = np.zeros((6, 8, 3))
    assert np.allclose(composite(fg, bg), fg)


def test_demo_disc_kept_and_screen_replaced():
    fg = make_foreground()
    bg = make_background()
    out = composite(fg, bg, tolerance=60.0)
    assert np.allclose(out[30, 40], (255.0, 0.0, 0.0))  # disc centre
    assert np.allclose(out[0, 0], bg[0, 0])              # corner = screen
    assert np.allclose(out[59, 79], bg[59, 79])


def test_replaced_count_matches_disc_area():
    fg = make_foreground()
    replaced = replaced_pixel_count(fg, tolerance=60.0)
    total = 60 * 80
    disc_area = int(np.pi * 15 ** 2)
    assert abs(replaced - (total - disc_area)) < 60  # antialias-free circle
    assert replaced > total // 2


def test_tolerance_boundary():
    # (0, 200, 0) is 55 from pure green; (0, 180, 0) is 75 away.
    fg = np.zeros((1, 2, 3))
    fg[0, 0] = (0.0, 200.0, 0.0)
    fg[0, 1] = (0.0, 180.0, 0.0)
    bg = np.full((1, 2, 3), 99.0)
    out = composite(fg, bg, tolerance=60.0)
    assert np.allclose(out[0, 0], 99.0)   # within tolerance: replaced
    assert np.allclose(out[0, 1], fg[0, 1])  # outside: kept


def test_custom_key_color():
    fg = np.zeros((4, 4, 3))
    fg[:, :, 2] = 255.0  # pure blue screen
    bg = np.full((4, 4, 3), 7.0)
    out = composite(fg, bg, key_color=(0.0, 0.0, 255.0), tolerance=10.0)
    assert np.allclose(out, bg)


def test_feather_gives_a_partial_blend():
    fg = np.zeros((1, 1, 3))
    fg[0, 0] = (0.0, 200.0, 0.0)  # distance 55 from green
    bg = np.full((1, 1, 3), 100.0)
    hard = composite(fg, bg, tolerance=60.0, feather=0.0)
    assert np.allclose(hard[0, 0], 100.0)  # hard key replaces it
    soft = composite(fg, bg, tolerance=50.0, feather=20.0)
    # alpha = (55 - 50) / 20 = 0.25 foreground, 0.75 background.
    assert soft[0, 0, 1] == pytest.approx(0.25 * 200.0 + 0.75 * 100.0)


def test_green_mask_values():
    fg = make_foreground()
    mask = green_mask(fg, tolerance=60.0)
    assert mask.shape == (60, 80)
    assert mask[30, 40] == 1.0   # disc: foreground
    assert mask[0, 0] == 0.0     # screen
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_color_distance_is_zero_on_key_colour():
    img = np.zeros((3, 3, 3))
    img[:, :, 1] = 255.0
    assert np.allclose(color_distance(img), 0.0)
    img[:, :, 1] = 0.0
    assert np.allclose(color_distance(img), 255.0)


def test_shape_mismatch_raises():
    fg = np.zeros((6, 8, 3))
    bg = np.zeros((6, 9, 3))
    with pytest.raises(ValueError):
        composite(fg, bg)


def test_grayscale_image_raises():
    with pytest.raises(ValueError):
        composite(np.zeros((6, 8)), np.zeros((6, 8, 3)))
    with pytest.raises(ValueError):
        green_mask(np.zeros((6, 8)))


def test_bad_key_color_raises():
    fg = np.zeros((4, 4, 3))
    with pytest.raises(ValueError):
        composite(fg, fg, key_color=(0, 255))
    with pytest.raises(ValueError):
        composite(fg, fg, key_color=(0, 255, 300))


def test_negative_tolerance_and_feather_raise():
    fg = np.zeros((4, 4, 3))
    with pytest.raises(ValueError):
        composite(fg, fg, tolerance=-1.0)
    with pytest.raises(ValueError):
        composite(fg, fg, feather=-5.0)


def test_empty_image_raises():
    with pytest.raises(ValueError):
        composite(np.zeros((0, 0, 3)), np.zeros((0, 0, 3)))
