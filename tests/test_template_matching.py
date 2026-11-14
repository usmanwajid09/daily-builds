import numpy as np
import pytest

from template_matching import match_template, match_template_map, ncc_at
from template_matching.demo import TRUE_COL, TRUE_ROW, make_test_image


def test_exact_copy_scores_one_at_true_location():
    image, template = make_test_image()
    row, col, score = match_template(image, template)
    assert (row, col) == (TRUE_ROW, TRUE_COL)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_score_map_shape_and_range():
    image, template = make_test_image()
    score_map = match_template_map(image, template)
    assert score_map.shape == (60 - 12 + 1, 80 - 10 + 1)
    assert score_map.dtype == np.float64
    assert score_map.min() >= -1.0
    assert score_map.max() <= 1.0
    assert np.isfinite(score_map).all()


def test_match_agrees_with_score_map_argmax():
    image, template = make_test_image()
    score_map = match_template_map(image, template)
    row, col, score = match_template(image, template)
    assert score == pytest.approx(score_map.max())
    assert score == pytest.approx(score_map[row, col])


def test_inverted_template_scores_minus_one():
    rng = np.random.default_rng(3)
    image = rng.random((20, 20)) * 255.0
    template = image[4:9, 6:12].copy()
    # Flip the patch in place: it now correlates -1 with the template.
    # (The inverted copy is the *worst* match, so check the score map
    # minimum and ncc_at, not the argmax.)
    image[4:9, 6:12] = 255.0 - template
    score_map = match_template_map(image, template)
    assert score_map[4, 6] == pytest.approx(-1.0, abs=1e-9)
    assert score_map.min() == pytest.approx(-1.0, abs=1e-9)
    assert ncc_at(image, template, 4, 6) == pytest.approx(-1.0, abs=1e-9)


def test_brightness_shift_does_not_change_score():
    image, template = make_test_image()
    bright = image.copy()
    bright[
        TRUE_ROW:TRUE_ROW + 12, TRUE_COL:TRUE_COL + 10
    ] = bright[TRUE_ROW:TRUE_ROW + 12, TRUE_COL:TRUE_COL + 10] * 0.5 + 40.0
    row, col, score = match_template(bright, template)
    assert (row, col) == (TRUE_ROW, TRUE_COL)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_constant_template_scores_zero_not_nan():
    image, _ = make_test_image()
    template = np.full((6, 6), 128.0)
    score_map = match_template_map(image, template)
    assert np.all(score_map == 0.0)
    assert np.isfinite(score_map).all()
    row, col, score = match_template(image, template)
    assert score == 0.0
    assert (row, col) == (0, 0)  # first position wins the tie


def test_constant_image_patch_scores_zero():
    image = np.full((20, 20), 50.0)
    image[5:10, 5:10] = np.arange(25).reshape(5, 5)
    template = np.arange(25).reshape(5, 5).astype(float)
    # The only exact match is the template itself; windows that do not
    # touch the non-constant block at all are flat and must score 0.
    # (Windows overlapping the block partially are non-constant, so they
    # get small nonzero scores.)
    score_map = match_template_map(image, template)
    assert score_map[5, 5] == pytest.approx(1.0, abs=1e-9)
    assert np.all(score_map[0, :] == 0.0)   # rows above the block
    assert np.all(score_map[10:, :] == 0.0)  # rows below the block
    assert np.all(score_map[:, 0] == 0.0)   # cols left of the block
    assert np.all(score_map[:, 10:] == 0.0)  # cols right of the block


def test_template_same_size_as_image():
    image = np.random.default_rng(5).random((8, 8))
    row, col, score = match_template(image, image.copy())
    assert (row, col) == (0, 0)
    assert score == pytest.approx(1.0, abs=1e-9)
    assert match_template_map(image, image.copy()).shape == (1, 1)


def test_ncc_at_matches_score_map():
    image, template = make_test_image()
    score_map = match_template_map(image, template)
    for row, col in ((0, 0), (TRUE_ROW, TRUE_COL), (10, 30), (48, 69)):
        assert ncc_at(image, template, row, col) == pytest.approx(
            score_map[row, col], abs=1e-12
        )


def test_ncc_at_out_of_bounds_raises():
    image, template = make_test_image()
    with pytest.raises(ValueError):
        ncc_at(image, template, 49, 0)  # 49 + 12 > 60
    with pytest.raises(ValueError):
        ncc_at(image, template, 0, 71)  # 71 + 10 > 80
    with pytest.raises(ValueError):
        ncc_at(image, template, -1, 0)


def test_uint8_input_works():
    image = (np.random.default_rng(7).random((30, 30)) * 255).astype(np.uint8)
    template = image[10:15, 10:15].copy()
    row, col, score = match_template(image, template)
    assert (row, col) == (10, 10)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_template_larger_than_image_raises():
    image = np.zeros((5, 5))
    with pytest.raises(ValueError):
        match_template(image, np.zeros((6, 5)))
    with pytest.raises(ValueError):
        match_template(image, np.zeros((5, 6)))
    with pytest.raises(ValueError):
        match_template_map(image, np.zeros((10, 10)))


def test_wrong_ndim_raises():
    image = np.zeros((10, 10))
    template = np.zeros((3, 3))
    with pytest.raises(ValueError):
        match_template(np.zeros((10,)), template)  # 1D image
    with pytest.raises(ValueError):
        match_template(image, np.zeros((3, 3, 3)))  # 3D template
    with pytest.raises(ValueError):
        match_template(np.zeros((10, 10, 3)), template)  # RGB image


def test_non_numeric_dtype_raises():
    image = np.zeros((10, 10))
    template = np.zeros((3, 3))
    with pytest.raises(ValueError):
        match_template(np.array([["a", "b"], ["c", "d"]]), template)
    with pytest.raises(ValueError):
        match_template(image, np.array([["a", "b"], ["c", "d"]]))


def test_empty_input_raises():
    with pytest.raises(ValueError):
        match_template(np.zeros((0, 5)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        match_template(np.zeros((5, 5)), np.zeros((0, 2)))


def test_non_finite_values_raise():
    image = np.zeros((10, 10))
    template = np.zeros((3, 3))
    bad_image = image.copy()
    bad_image[2, 2] = np.nan
    with pytest.raises(ValueError):
        match_template(bad_image, template)
    bad_template = template.copy()
    bad_template[0, 0] = np.inf
    with pytest.raises(ValueError):
        match_template(image, bad_template)
