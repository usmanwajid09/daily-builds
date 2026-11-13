import numpy as np
import pytest

from morphological_ops.demo import make_test_image
from morphological_ops.morphology import (
    closing,
    dilate,
    erode,
    opening,
    structuring_element,
)


def _rect():
    img = np.zeros((10, 10), dtype=bool)
    img[3:7, 3:7] = True  # 4x4 solid square
    return img


def test_erosion_shrinks_solid_square():
    out = erode(_rect())
    assert out.sum() == 4  # 4x4 -> 2x2 with a 3x3 element
    assert out[4:6, 4:6].all()
    assert not out[3, :].any()


def test_dilation_grows_solid_square():
    out = dilate(_rect())
    assert out.sum() == 36  # 4x4 -> 6x6 with a 3x3 element
    assert out[2:8, 2:8].all()


def test_opening_removes_isolated_pixel():
    img = _rect()
    img[0, 0] = True  # isolated speck, far from the square
    out = opening(img)
    assert not out[0, 0]
    assert out.sum() == 16  # square survives opening intact


def test_closing_fills_single_hole():
    img = _rect()
    img[5, 5] = False  # one-pixel hole inside the square
    out = closing(img)
    assert out[5, 5]
    assert out.sum() == 16


def test_opening_and_closing_are_idempotent():
    img = make_test_image()
    assert np.array_equal(opening(opening(img)), opening(img))
    assert np.array_equal(closing(closing(img)), closing(img))


def test_demo_image_opening_removes_noise():
    img = make_test_image()
    opened = opening(img)
    # Noise specks gone, rectangle (192 px minus its 1-px hole) kept.
    assert opened.sum() == 191
    assert opened.sum() < img.sum()


def test_cross_element_is_smaller_than_square():
    img = _rect()
    square = erode(img, structuring_element("square", 3))
    cross = erode(img, structuring_element("cross", 3))
    # The cross touches fewer neighbours, so it erodes less.
    assert cross.sum() >= square.sum()
    assert dilate(img, structuring_element("cross", 3)).sum() <= dilate(img).sum()


def test_custom_element_respected():
    # A vertical 3x1 element only erodes top/bottom, not left/right.
    element = np.array([[True], [True], [True]])
    img = _rect()
    out = erode(img, element)
    assert out.sum() == 8  # 4x4 -> 2 rows x 4 cols
    assert out[4:6, 3:7].all()


def test_numeric_01_input_accepted():
    img = _rect().astype(np.uint8)
    assert erode(img).sum() == 4
    assert dilate(img).sum() == 36


def test_output_shape_and_dtype():
    img = np.random.default_rng(0).random((7, 11)) > 0.5
    for op in (erode, dilate, opening, closing):
        out = op(img)
        assert out.shape == img.shape
        assert out.dtype == bool


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        erode(np.zeros((5, 5, 3)))  # not 2D
    with pytest.raises(ValueError):
        erode(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        erode(np.array([[0, 2], [1, 0]]))  # values other than 0/1
    with pytest.raises(ValueError):
        erode(np.array([["a", "b"]]))  # non-numeric
    with pytest.raises(ValueError):
        erode(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        erode(_rect(), np.ones((2, 2), dtype=bool))  # even element
    with pytest.raises(ValueError):
        erode(_rect(), np.zeros((3, 3), dtype=bool))  # all-False element
    with pytest.raises(ValueError):
        structuring_element("diamond")
    with pytest.raises(ValueError):
        structuring_element("square", 4)
    with pytest.raises(ValueError):
        structuring_element("square", 0)
