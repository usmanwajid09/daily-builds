import numpy as np
import pytest

from zhang_suen_thinning.demo import make_test_image
from zhang_suen_thinning.thinning import (
    is_skeleton,
    skeleton_pixel_count,
    skeletonize,
    zhang_suen_thin,
)


def test_3x3_block_thins_to_centre_pixel():
    img = np.zeros((7, 7), dtype=bool)
    img[2:5, 2:5] = True  # a 3x3 block in the middle of a 7x7 canvas
    out = zhang_suen_thin(img)
    assert out.sum() == 1
    assert out[3, 3]


def test_thick_horizontal_line_thins_to_one_row():
    img = np.zeros((7, 11), dtype=bool)
    img[3:6, 1:10] = True  # 3 px thick, 9 px long
    out = zhang_suen_thin(img)
    assert out[4, 2:8].all()
    assert not out[3].any() and not out[5].any()
    # Zhang-Suen erodes the blunt 3 px-wide ends a little, so the
    # centreline is shorter than the bar (6 of the 9 columns).
    assert out.sum() == 6


def test_thick_vertical_line_thins_to_one_column():
    img = np.zeros((11, 7), dtype=bool)
    img[1:10, 3:6] = True  # 3 px thick, 9 px long
    out = zhang_suen_thin(img)
    assert out[2:8, 4].all()
    assert not out[:, 3].any() and not out[:, 5].any()
    # As with the horizontal bar, the blunt ends erode a little.
    assert out.sum() == 6


def test_single_pixel_survives():
    img = np.zeros((5, 5), dtype=bool)
    img[2, 2] = True
    assert np.array_equal(zhang_suen_thin(img), img)


def test_empty_image_stays_empty():
    img = np.zeros((6, 6), dtype=bool)
    assert not zhang_suen_thin(img).any()


def test_thin_line_is_unchanged():
    img = np.zeros((7, 7), dtype=bool)
    img[3, 1:6] = True
    assert np.array_equal(zhang_suen_thin(img), img)
    assert is_skeleton(img)


def test_skeleton_is_subset_of_input():
    img = make_test_image()
    out = zhang_suen_thin(img)
    assert (out & ~img).sum() == 0
    assert out.sum() < img.sum()
    assert out.sum() > 0


def test_thinning_is_idempotent():
    img = make_test_image()
    once = zhang_suen_thin(img)
    twice = zhang_suen_thin(once)
    assert np.array_equal(once, twice)
    assert is_skeleton(once)


def test_plus_sign_keeps_its_centreline():
    img = make_test_image()
    out = zhang_suen_thin(img)
    assert out[10, 14]  # centre of the plus survives
    assert out[10, 5:25].sum() >= 15  # horizontal arm stays long
    assert out[3:17, 14].sum() >= 10  # vertical arm stays long


def test_0_255_input_is_accepted():
    img = np.zeros((7, 11), dtype=np.uint8)
    img[3:6, 1:10] = 255
    out = zhang_suen_thin(img)
    assert out.dtype == bool
    assert out.sum() == 6


def test_skeletonize_alias_matches():
    img = make_test_image()
    assert np.array_equal(skeletonize(img), zhang_suen_thin(img))


def test_skeleton_pixel_count():
    img = make_test_image()
    assert skeleton_pixel_count(img) == int(img.sum())
    assert skeleton_pixel_count(zhang_suen_thin(img)) < int(img.sum())


def test_non_binary_values_raise():
    img = np.zeros((5, 5))
    img[2, 2] = 0.5
    with pytest.raises(ValueError):
        zhang_suen_thin(img)
    img[2, 2] = 2.0
    with pytest.raises(ValueError):
        zhang_suen_thin(img)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        zhang_suen_thin(np.zeros((5, 5, 3)))
    with pytest.raises(ValueError):
        zhang_suen_thin(np.zeros((5,)))
    with pytest.raises(ValueError):
        zhang_suen_thin(np.zeros((0, 0)))
    with pytest.raises(ValueError):
        skeleton_pixel_count(np.zeros((5, 5, 2)))
