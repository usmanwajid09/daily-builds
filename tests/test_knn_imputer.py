"""Tests for the from-scratch kNN imputer."""
import numpy as np
import pytest

from knn_imputer import KNNImputer


@pytest.fixture
def exact():
    # Query row [nan, 15]: distances to the two donors are equal
    # (both |15 - 10| and |15 - 20| are 5 on the one shared feature),
    # so the uniform mean of their feature-0 values is exactly (1 + 3) / 2.
    return np.array([[1.0, 10.0], [3.0, 20.0], [np.nan, 15.0]])


@pytest.fixture
def weighted():
    # Query row [nan, 11]: donor 0 is much closer (|11-10|=1) than donor 1
    # (|11-20|=9), so distance weighting gives exactly
    # (1 + 3/9) / (1 + 1/9) = 1.2, while uniform gives (1 + 3) / 2 = 2.0.
    return np.array([[1.0, 10.0], [3.0, 20.0], [np.nan, 11.0]])


def test_no_missing_passthrough_returns_float_copy():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = KNNImputer().fit_transform(X)
    np.testing.assert_array_equal(out, X)
    assert out.dtype == np.float64
    assert out is not X
    out[0, 0] = 99.0
    assert X[0, 0] == 1.0  # input not mutated


def test_uniform_imputation_exact_hand_computed(exact):
    out = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(exact)
    assert out[2, 0] == pytest.approx(2.0)
    # Rows without missing values are untouched.
    np.testing.assert_allclose(out[:2], exact[:2])


def test_distance_weighted_imputation_exact_hand_computed(weighted):
    out = KNNImputer(n_neighbors=2, weights="distance").fit_transform(weighted)
    assert out[2, 0] == pytest.approx(1.2)


def test_distance_weighted_differs_from_uniform(weighted):
    uniform = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(weighted)
    distance = KNNImputer(n_neighbors=2, weights="distance").fit_transform(weighted)
    assert uniform[2, 0] == pytest.approx(2.0)
    assert distance[2, 0] == pytest.approx(1.2)
    assert uniform[2, 0] != distance[2, 0]


def test_all_missing_column_raises_at_fit():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    with pytest.raises(ValueError, match="column 1 is entirely missing"):
        KNNImputer().fit(X)


def test_transform_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        KNNImputer().transform([[1.0, np.nan]])


def test_n_neighbors_zero_raises():
    with pytest.raises(ValueError, match="n_neighbors"):
        KNNImputer(n_neighbors=0)


def test_bad_weights_raises():
    with pytest.raises(ValueError, match="weights"):
        KNNImputer(weights="inverse")


def test_1d_input_raises():
    with pytest.raises(ValueError, match="2D"):
        KNNImputer().fit([1.0, np.nan, 3.0])


def test_inf_raises():
    with pytest.raises(ValueError, match="inf"):
        KNNImputer().fit([[1.0, np.inf], [2.0, 3.0]])
    model = KNNImputer().fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="inf"):
        model.transform([[1.0, np.inf]])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        KNNImputer().fit(np.empty((0, 2)))


def test_feature_count_mismatch_on_transform_raises():
    model = KNNImputer().fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="features"):
        model.transform([[1.0, 2.0, np.nan]])


def test_output_shape_and_dtype_preserved():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    X[rng.random(X.shape) < 0.3] = np.nan
    out = KNNImputer(n_neighbors=3).fit_transform(X)
    assert out.shape == X.shape
    assert out.dtype == np.float64


def test_no_nan_remains_in_output():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    X[rng.random(X.shape) < 0.3] = np.nan
    out = KNNImputer(n_neighbors=3).fit_transform(X)
    assert not np.isnan(out).any()


def test_all_missing_row_gets_column_means():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [np.nan, np.nan]])
    out = KNNImputer(n_neighbors=2).fit_transform(X)
    np.testing.assert_allclose(out[2], [2.0, 15.0])


def test_donors_missing_the_feature_are_skipped():
    # Row 1 is closest to the query on feature 1 but has no feature-0 value,
    # so the imputation must average rows 0 and 2 only: (1 + 5) / 2 = 3.0.
    X = np.array(
        [[1.0, 10.0], [np.nan, 11.0], [5.0, 20.0], [np.nan, 12.0]]
    )
    out = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(X)
    assert out[3, 0] == pytest.approx(3.0)


def test_column_mean_fallback_when_no_donor_observed_feature():
    # Only row 1 observes feature 1, and it does not observe feature 0;
    # the query row shares no observed feature with row 0, so the only
    # usable donor for feature 0 is none -> column mean 1.0.
    X = np.array([[1.0, np.nan], [np.nan, 5.0]])
    out = KNNImputer(n_neighbors=1).fit_transform(X)
    np.testing.assert_allclose(out, [[1.0, 5.0], [1.0, 5.0]])


def test_deterministic_same_input_twice(exact):
    model = KNNImputer(n_neighbors=2, weights="distance").fit(exact)
    first = model.transform(exact)
    second = model.transform(exact)
    np.testing.assert_array_equal(first, second)


def test_k_larger_than_available_donors_works(exact):
    out = KNNImputer(n_neighbors=10, weights="uniform").fit_transform(exact)
    assert out[2, 0] == pytest.approx(2.0)  # uses all 2 available donors
    assert not np.isnan(out).any()


def test_fit_transform_matches_fit_then_transform(exact):
    combined = KNNImputer(n_neighbors=2).fit_transform(exact)
    model = KNNImputer(n_neighbors=2).fit(exact)
    separate = model.transform(exact)
    np.testing.assert_array_equal(combined, separate)
