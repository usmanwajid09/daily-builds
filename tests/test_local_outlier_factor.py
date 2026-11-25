"""Tests for local_outlier_factor (from scratch, numpy only)."""

import numpy as np
import pytest

from local_outlier_factor.model import LocalOutlierFactor


def make_blob_with_outliers(seed=42, n_in=100, n_out=5):
    """Dense Gaussian blob plus far-away planted outliers."""
    rng = np.random.default_rng(seed)
    X_in = rng.normal(loc=0.0, scale=0.5, size=(n_in, 2))
    X_out = np.array([[8.0, 8.0], [-8.0, 7.0], [7.0, -8.0],
                      [-7.0, -7.0], [10.0, 0.0]])
    return np.vstack([X_in, X_out]), n_in, n_out


def test_planted_outliers_flagged():
    X, n_in, _ = make_blob_with_outliers()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    labels = lof.fit_predict(X)
    assert (labels[n_in:] == -1).all(), "planted outliers not flagged -1"


def test_outlier_lof_greater_than_inlier_lof():
    X, n_in, _ = make_blob_with_outliers()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit(X)
    lof_scores = -lof.negative_outlier_factor_
    assert lof_scores[n_in:].min() > lof_scores[:n_in].max()


def test_uniform_data_lof_near_one():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200, 2))
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit(X)
    lof_scores = -lof.negative_outlier_factor_
    # Uniform data: no real outliers, LOF should cluster around 1.
    assert np.median(lof_scores) == pytest.approx(1.0, abs=0.3)
    assert lof_scores.max() < 3.0


def test_fit_predict_labels_shape_and_values():
    X, _, _ = make_blob_with_outliers()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    labels = lof.fit_predict(X)
    assert labels.shape == (len(X),)
    assert set(np.unique(labels).tolist()) <= {1, -1}
    assert (labels == -1).sum() >= 1


def test_decision_function_higher_for_inliers():
    X, n_in, _ = make_blob_with_outliers()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit(X)
    scores = lof.decision_function(X)
    assert scores.shape == (len(X),)
    assert scores[:n_in].mean() > scores[n_in:].mean()
    assert (scores[n_in:] < 0).all()


def test_score_samples_consistent_with_negative_outlier_factor():
    X, _, _ = make_blob_with_outliers()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit(X)
    scores = lof.score_samples(X)
    np.testing.assert_allclose(scores, lof.negative_outlier_factor_, rtol=1e-10)


def test_contamination_changes_flagged_count():
    X, _, _ = make_blob_with_outliers()
    low = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit(X)
    high = LocalOutlierFactor(n_neighbors=20, contamination=0.3).fit(X)
    assert (high.labels_ == -1).sum() > (low.labels_ == -1).sum()


def test_single_cluster_point_handling():
    # One isolated point among a tight cluster: flagged, no crash.
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, size=(50, 2)), [[5.0, 5.0]]])
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.05).fit(X)
    assert lof.labels_[-1] == -1
    assert np.isfinite(lof.negative_outlier_factor_).all()


def test_empty_input_raises():
    with pytest.raises(ValueError):
        LocalOutlierFactor().fit(np.empty((0, 2)))


def test_1d_input_raises():
    with pytest.raises(ValueError):
        LocalOutlierFactor().fit(np.array([1.0, 2.0, 3.0]))


def test_n_neighbors_ge_n_samples_raises():
    X = np.random.default_rng(2).normal(size=(10, 2))
    with pytest.raises(ValueError):
        LocalOutlierFactor(n_neighbors=10).fit(X)
    with pytest.raises(ValueError):
        LocalOutlierFactor(n_neighbors=20).fit(X)


def test_nan_raises():
    X = np.random.default_rng(3).normal(size=(30, 2))
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        LocalOutlierFactor().fit(X)


def test_inf_raises():
    X = np.random.default_rng(3).normal(size=(30, 2))
    X[5, 1] = np.inf
    with pytest.raises(ValueError):
        LocalOutlierFactor().fit(X)


def test_predict_before_fit_raises():
    lof = LocalOutlierFactor()
    X = np.random.default_rng(4).normal(size=(30, 2))
    with pytest.raises(RuntimeError):
        lof.predict(X)
    with pytest.raises(RuntimeError):
        lof.decision_function(X)


def test_contamination_out_of_range_raises():
    with pytest.raises(ValueError):
        LocalOutlierFactor(contamination=0.0)
    with pytest.raises(ValueError):
        LocalOutlierFactor(contamination=0.6)
    with pytest.raises(ValueError):
        LocalOutlierFactor(contamination=-0.1)


def test_predict_new_points():
    X, n_in, _ = make_blob_with_outliers()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit(X)
    new_inliers = np.array([[0.1, 0.0], [-0.2, 0.3]])
    new_outliers = np.array([[9.0, 9.0], [-9.0, -9.0]])
    assert (lof.predict(new_inliers) == 1).all()
    assert (lof.predict(new_outliers) == -1).all()


def test_deterministic():
    X, _, _ = make_blob_with_outliers()
    a = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit(X)
    b = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit(X)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_allclose(a.negative_outlier_factor_,
                               b.negative_outlier_factor_)
