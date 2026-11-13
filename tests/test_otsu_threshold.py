import numpy as np
import pytest

from otsu_threshold.demo import make_test_image
from otsu_threshold.otsu import apply_threshold, binarize, otsu_threshold


def test_bimodal_threshold_lies_between_modes():
    img = make_test_image()
    threshold = otsu_threshold(img)
    # Modes are ~50 and ~200; the split should sit in the valley.
    assert 80 <= threshold <= 170


def test_bimodal_foreground_share_is_about_half():
    mask, threshold = binarize(make_test_image())
    share = mask.mean()
    assert 0.45 <= share <= 0.55


def test_two_value_image_threshold_between_values():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:, 5:] = 200
    threshold = otsu_threshold(img)
    assert 0 <= threshold < 200
    mask = apply_threshold(img, threshold)
    assert mask[:, :5].sum() == 0
    assert mask[:, 5:].sum() == 50


def test_constant_image_returns_its_value():
    assert otsu_threshold(np.full((5, 5), 42, dtype=np.uint8)) == 42
    assert otsu_threshold(np.zeros((5, 5))) == 0
    assert otsu_threshold(np.full((5, 5), 255, dtype=np.uint8)) == 255


def test_threshold_is_deterministic():
    rng = np.random.default_rng(3)
    img = (rng.normal(100, 40, size=(30, 30))).clip(0, 255)
    assert otsu_threshold(img) == otsu_threshold(img)


def test_apply_threshold_returns_bool_mask():
    img = np.array([[0, 100, 200]], dtype=np.uint8)
    mask = apply_threshold(img, 100)
    assert mask.dtype == bool
    assert mask.tolist() == [[False, False, True]]
    assert apply_threshold(img, 0).tolist() == [[False, True, True]]
    assert apply_threshold(img, 255).tolist() == [[False, False, False]]


def test_binarize_agrees_with_threshold_and_mask():
    img = make_test_image()
    mask, threshold = binarize(img)
    assert threshold == otsu_threshold(img)
    assert np.array_equal(mask, apply_threshold(img, threshold))


def test_output_shape_matches_input():
    img = np.random.default_rng(0).integers(0, 256, size=(7, 11)).astype(np.uint8)
    mask, _ = binarize(img)
    assert mask.shape == img.shape


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros((5, 5, 3)))  # RGB not supported here
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros((5,)))  # 1D
    with pytest.raises(ValueError):
        otsu_threshold(np.zeros((0, 5)))  # empty
    with pytest.raises(ValueError):
        otsu_threshold(np.full((5, 5), np.nan))  # non-finite
    with pytest.raises(ValueError):
        otsu_threshold(np.array([["a", "b"]]))  # non-numeric
    with pytest.raises(ValueError):
        apply_threshold(np.zeros((5, 5)), -1)
    with pytest.raises(ValueError):
        apply_threshold(np.zeros((5, 5)), 256)
    with pytest.raises(ValueError):
        apply_threshold(np.zeros((5, 5)), 127.5)  # must be an integer
