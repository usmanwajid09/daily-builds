"""Tests for knn_anomaly_detector.KNNAnomalyDetector."""
import numpy as np
import pytest

from knn_anomaly_detector.model import KNNAnomalyDetector


def make_blob(seed=0, n=200, dim=2, scale=1.0, shift=0.0):
    rng = np.random.default_rng(seed)
    return rng.normal(shift, scale, size=(n, dim))


def test_detects_far_outliers():
    x_train = make_blob(seed=0)
    rng = np.random.default_rng(1)
    x_outlier = rng.normal(0, 1, size=(30, 2)) + np.array([8.0, 8.0])
    model = KNNAnomalyDetector().fit(x_train)
    assert float(np.mean(model.predict(x_outlier) == -1)) >= 0.9


def test_fresh_inlier_rate_matches_contamination():
    x_train = make_blob(seed=2)
    x_fresh = make_blob(seed=19, n=500)
    model = KNNAnomalyDetector(contamination=0.1).fit(x_train)
    flagged = float(np.mean(model.predict(x_fresh) == -1))
    assert 0.03 <= flagged <= 0.20


def test_train_points_flagged_no_more_than_contamination():
    # Train points are scored with themselves included, so their
    # scores are lower than the calibrated leave-one-out scores.
    x_train = make_blob(seed=2)
    model = KNNAnomalyDetector(contamination=0.1).fit(x_train)
    flagged = float(np.mean(model.predict(x_train) == -1))
    assert flagged <= 0.15


def test_fresh_inliers_mostly_normal():
    x_train = make_blob(seed=3)
    x_fresh = make_blob(seed=4, n=100)
    model = KNNAnomalyDetector().fit(x_train)
    assert float(np.mean(model.predict(x_fresh) == 1)) >= 0.8


def test_scores_ordered():
    x_train = make_blob(seed=5)
    rng = np.random.default_rng(6)
    x_normal = make_blob(seed=7, n=50)
    x_outlier = rng.normal(0, 1, size=(50, 2)) + np.array([8.0, 8.0])
    model = KNNAnomalyDetector().fit(x_train)
    assert model.anomaly_score(x_normal).mean() < model.anomaly_score(x_outlier).mean()
    assert model.score_samples(x_normal).mean() > model.score_samples(x_outlier).mean()
    assert model.decision_function(x_normal).mean() > model.decision_function(x_outlier).mean()


def test_predict_returns_only_plus_minus_one():
    x_train = make_blob(seed=8)
    model = KNNAnomalyDetector().fit(x_train)
    pred = model.predict(x_train)
    assert set(np.unique(pred)) <= {1, -1}
    assert pred.shape == (len(x_train),)


def test_threshold_is_quantile_of_train_scores():
    x_train = make_blob(seed=9)
    model = KNNAnomalyDetector(contamination=0.1).fit(x_train)
    expected = float(np.quantile(model.train_scores_, 0.9))
    assert model.threshold_ == pytest.approx(expected)
    assert model.train_scores_.shape == (len(x_train),)


def test_higher_contamination_flags_more():
    x_train = make_blob(seed=10)
    low = KNNAnomalyDetector(contamination=0.05).fit(x_train)
    high = KNNAnomalyDetector(contamination=0.3).fit(x_train)
    assert high.threshold_ < low.threshold_
    assert float(np.mean(high.predict(x_train) == -1)) > float(np.mean(low.predict(x_train) == -1))


def test_larger_k_smooths_scores():
    x_train = make_blob(seed=11)
    small = KNNAnomalyDetector(n_neighbors=1).fit(x_train)
    large = KNNAnomalyDetector(n_neighbors=20).fit(x_train)
    # With k=1 the train scores are much smaller (nearest neighbour only).
    assert small.train_scores_.mean() < large.train_scores_.mean()


def test_manhattan_metric():
    x_train = make_blob(seed=12)
    rng = np.random.default_rng(13)
    x_outlier = rng.normal(0, 1, size=(20, 2)) + np.array([8.0, 8.0])
    model = KNNAnomalyDetector(metric="manhattan").fit(x_train)
    assert float(np.mean(model.predict(x_outlier) == -1)) >= 0.9
    assert model.anomaly_score(x_train).mean() > 0


def test_exact_duplicate_points_score_zero():
    x_train = np.zeros((10, 2))
    model = KNNAnomalyDetector(n_neighbors=3).fit(x_train)
    assert np.allclose(model.anomaly_score(x_train), 0.0)
    assert np.all(model.predict(x_train) == 1)


def test_1d_data():
    x_train = make_blob(seed=14, dim=1, n=150)
    rng = np.random.default_rng(15)
    x_outlier = rng.normal(0, 1, size=(20, 1)) + 8.0
    model = KNNAnomalyDetector().fit(x_train)
    assert float(np.mean(model.predict(x_outlier) == -1)) >= 0.9


def test_predict_before_fit_raises():
    model = KNNAnomalyDetector()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((2, 2)))


def test_value_error_1d_X():
    with pytest.raises(ValueError, match="2D"):
        KNNAnomalyDetector().fit(np.zeros(10))


def test_value_error_empty_input():
    with pytest.raises(ValueError, match="empty"):
        KNNAnomalyDetector().fit(np.zeros((0, 2)))


def test_value_error_nonfinite():
    x = make_blob(seed=16)
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        KNNAnomalyDetector().fit(x)


def test_value_error_too_few_samples():
    x = make_blob(seed=17, n=5)
    with pytest.raises(ValueError, match="more than n_neighbors"):
        KNNAnomalyDetector(n_neighbors=5).fit(x)


def test_value_error_bad_params():
    with pytest.raises(ValueError):
        KNNAnomalyDetector(n_neighbors=0)
    with pytest.raises(ValueError):
        KNNAnomalyDetector(contamination=0)
    with pytest.raises(ValueError):
        KNNAnomalyDetector(contamination=0.6)
    with pytest.raises(ValueError):
        KNNAnomalyDetector(metric="cosine")


def test_value_error_wrong_feature_count_at_predict():
    x_train = make_blob(seed=18)
    model = KNNAnomalyDetector().fit(x_train)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 5)))
