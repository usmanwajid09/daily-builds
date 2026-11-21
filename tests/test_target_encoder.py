"""Tests for TargetEncoder (smoothed target encoding)."""

import numpy as np
import pytest

from target_encoder import TargetEncoder


def test_smoothing_zero_gives_raw_means():
    X = np.array(["a", "a", "b", "b", "b"])
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    enc = TargetEncoder(smoothing=0.0).fit(X, y)
    assert enc.transform(np.array(["a"]))[0] == pytest.approx(0.5)
    assert enc.transform(np.array(["b"]))[0] == pytest.approx(2 / 3)


def test_formula_exact_hand_computed():
    # global mean = (1+0+1+1)/4 = 0.75; "a": count 2, mean 0.5
    # enc(a) = (2*0.5 + 1*0.75) / (2+1) = 1.75/3
    X = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    enc = TargetEncoder(smoothing=1.0).fit(X, y)
    assert enc.global_mean_ == pytest.approx(0.75)
    assert enc.transform(np.array(["a"]))[0] == pytest.approx(1.75 / 3)
    assert enc.transform(np.array(["b"]))[0] == pytest.approx(2.75 / 3)


def test_large_smoothing_pulls_toward_global_mean():
    X = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    enc = TargetEncoder(smoothing=1000.0).fit(X, y)
    assert enc.transform(np.array(["a"]))[0] == pytest.approx(0.75, abs=0.01)
    assert enc.transform(np.array(["b"]))[0] == pytest.approx(0.75, abs=0.01)


def test_rare_category_shrinks_more_than_common_one():
    # Same raw mean (1.0) but n=1 vs n=100: the rare one must sit closer
    # to the global mean.
    X = np.array(["rare"] + ["common"] * 100 + ["other"] * 50)
    y = np.array([1.0] + [1.0] * 100 + [0.0] * 50)
    enc = TargetEncoder(smoothing=10.0).fit(X, y)
    gm = enc.global_mean_
    rare = enc.transform(np.array(["rare"]))[0]
    common = enc.transform(np.array(["common"]))[0]
    assert abs(rare - gm) < abs(common - gm)
    assert rare == pytest.approx((1 + 10 * gm) / 11)
    assert common == pytest.approx((100 + 10 * gm) / 110)


def test_unseen_category_encodes_to_global_mean():
    X = np.array(["a", "b", "a"])
    y = np.array([1.0, 0.0, 1.0])
    enc = TargetEncoder(smoothing=1.0).fit(X, y)
    assert enc.transform(np.array(["zzz"]))[0] == pytest.approx(
        enc.global_mean_)


def test_unseen_category_raises_with_error_mode():
    X = np.array(["a", "b", "a"])
    y = np.array([1.0, 0.0, 1.0])
    enc = TargetEncoder(handle_unknown="error").fit(X, y)
    with pytest.raises(ValueError, match="Unseen category"):
        enc.transform(np.array(["zzz"]))


def test_min_samples_leaf_maps_tiny_category_to_global_mean():
    X = np.array(["tiny", "big", "big", "big", "big", "big"])
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    enc = TargetEncoder(smoothing=0.0, min_samples_leaf=2).fit(X, y)
    assert enc.transform(np.array(["tiny"]))[0] == pytest.approx(
        enc.global_mean_)
    # "big" has 5 samples, so smoothing=0 gives its raw mean (2/5 = 0.4).
    assert enc.transform(np.array(["big"]))[0] == pytest.approx(0.4)


def test_fit_transform_equals_fit_then_transform():
    X = np.array(["a", "b", "a", "c", "b"])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    enc1 = TargetEncoder(smoothing=2.0)
    out1 = enc1.fit_transform(X, y)
    enc2 = TargetEncoder(smoothing=2.0).fit(X, y)
    out2 = enc2.transform(X)
    np.testing.assert_allclose(out1, out2)


def test_multi_column_support_and_shape():
    X = np.array([["a", "x"], ["b", "x"], ["a", "y"], ["b", "y"]])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    enc = TargetEncoder(smoothing=0.0).fit(X, y)
    out = enc.transform(X)
    assert out.shape == (4, 2)
    # Column 0: a -> 1.0, b -> 0.0. Column 1: x -> 0.5, y -> 0.5.
    np.testing.assert_allclose(out, [[1.0, 0.5], [0.0, 0.5],
                                     [1.0, 0.5], [0.0, 0.5]])


def test_2d_single_column_input():
    X = np.array([["a"], ["b"], ["a"]])
    y = np.array([1.0, 0.0, 1.0])
    enc = TargetEncoder(smoothing=0.0).fit(X, y)
    out = enc.transform(X)
    assert out.shape == (3, 1)
    np.testing.assert_allclose(out.ravel(), [1.0, 0.0, 1.0])


def test_transform_before_fit_raises():
    enc = TargetEncoder()
    with pytest.raises(ValueError, match="not been fitted"):
        enc.transform(np.array(["a"]))


def test_empty_input_raises():
    enc = TargetEncoder()
    with pytest.raises(ValueError, match="empty"):
        enc.fit(np.array([]), np.array([]))


def test_mismatched_lengths_raises():
    enc = TargetEncoder()
    with pytest.raises(ValueError, match="mismatched"):
        enc.fit(np.array(["a", "b"]), np.array([1.0]))


def test_y_with_nan_raises():
    enc = TargetEncoder()
    with pytest.raises(ValueError, match="NaN"):
        enc.fit(np.array(["a", "b"]), np.array([1.0, np.nan]))


def test_y_non_numeric_raises():
    enc = TargetEncoder()
    with pytest.raises(ValueError, match="numeric"):
        enc.fit(np.array(["a", "b"]), np.array(["yes", "no"]))


def test_negative_smoothing_raises():
    with pytest.raises(ValueError, match="smoothing"):
        TargetEncoder(smoothing=-1.0)


def test_min_samples_leaf_below_one_raises():
    with pytest.raises(ValueError, match="min_samples_leaf"):
        TargetEncoder(min_samples_leaf=0)


def test_bad_handle_unknown_raises():
    with pytest.raises(ValueError, match="handle_unknown"):
        TargetEncoder(handle_unknown="drop")


def test_bad_handle_missing_raises():
    with pytest.raises(ValueError, match="handle_missing"):
        TargetEncoder(handle_missing="drop")


def test_wrong_column_count_on_transform_raises():
    X = np.array([["a", "x"], ["b", "y"]])
    y = np.array([1.0, 0.0])
    enc = TargetEncoder().fit(X, y)
    with pytest.raises(ValueError, match="column"):
        enc.transform(np.array([["a"]]))


def test_global_mean_and_categories_stored():
    X = np.array(["b", "a", "b", "c"])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    enc = TargetEncoder().fit(X, y)
    assert enc.global_mean_ == pytest.approx(0.75)
    assert list(enc.categories_[0]) == ["b", "a", "c"]
    assert set(enc.encodings_[0]) == {"a", "b", "c"}


def test_string_and_int_categories_work():
    X = np.array([1, 2, 1, 2, 1])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    enc = TargetEncoder(smoothing=0.0).fit(X, y)
    assert enc.transform(np.array([1]))[0] == pytest.approx(30.0)
    assert enc.transform(np.array([2]))[0] == pytest.approx(30.0)
    assert enc.global_mean_ == pytest.approx(30.0)


def test_continuous_target():
    X = np.array(["a", "a", "b"])
    y = np.array([2.0, 4.0, 10.0])
    enc = TargetEncoder(smoothing=0.0).fit(X, y)
    assert enc.transform(np.array(["a"]))[0] == pytest.approx(3.0)
    assert enc.transform(np.array(["b"]))[0] == pytest.approx(10.0)


def test_missing_treated_as_own_category():
    X = np.array(["a", None, "a", None], dtype=object)
    y = np.array([1.0, 0.0, 1.0, 0.0])
    enc = TargetEncoder(smoothing=0.0).fit(X, y)
    assert enc.transform(np.array([None], dtype=object))[0] == pytest.approx(0.0)
    assert enc.transform(np.array(["a"]))[0] == pytest.approx(1.0)


def test_missing_raises_with_error_mode():
    X = np.array(["a", None, "a"], dtype=object)
    y = np.array([1.0, 0.0, 1.0])
    enc = TargetEncoder(handle_missing="error")
    with pytest.raises(ValueError, match="Missing value"):
        enc.fit(X, y)
