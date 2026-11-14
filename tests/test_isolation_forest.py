"""Tests for the isolation forest anomaly detector."""
import numpy as np
import pytest

from isolation_forest.model import IsolationForest


def make_data(seed=0, n_inliers=200, n_outliers=10):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(loc=0.0, scale=1.0, size=(n_inliers, 2))
    angles = rng.uniform(0, 2 * np.pi, size=n_outliers)
    radii = rng.uniform(8, 10, size=n_outliers)
    outliers = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([inliers, outliers]), inliers, outliers


def test_planted_outliers_score_higher_than_inliers():
    x, inliers, outliers = make_data()
    model = IsolationForest(contamination=0.05, random_state=0).fit(x)
    scores = model.decision_function(x)
    mean_inlier = scores[: len(inliers)].mean()
    mean_outlier = scores[len(inliers):].mean()
    assert mean_outlier > mean_inlier + 0.1


def test_predict_flags_most_planted_outliers():
    x, inliers, outliers = make_data()
    model = IsolationForest(contamination=0.05, random_state=0).fit(x)
    preds = model.predict(x)
    caught = np.sum(preds[len(inliers):] == -1)
    assert caught >= 8  # at least 8 of the 10 planted outliers


def test_inlier_heavy_data_mostly_predicted_inlier():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    model = IsolationForest(contamination=0.05, random_state=0).fit(x)
    preds = model.predict(x)
    assert np.mean(preds == 1) >= 0.90


def test_decision_function_shape_and_range():
    x, _, _ = make_data()
    model = IsolationForest(contamination=0.05, random_state=0).fit(x)
    scores = model.decision_function(x)
    assert scores.shape == (len(x),)
    assert np.all(scores > 0) and np.all(scores <= 1)


def test_deterministic_with_same_random_state():
    x, _, _ = make_data()
    a = IsolationForest(random_state=42).fit(x)
    b = IsolationForest(random_state=42).fit(x)
    assert np.allclose(a.decision_function(x), b.decision_function(x))
    assert np.array_equal(a.predict(x), b.predict(x))


def test_threshold_set_after_fit():
    x, _, _ = make_data()
    model = IsolationForest(contamination=0.05, random_state=0).fit(x)
    assert model.threshold_ is not None
    assert 0 < model.threshold_ <= 1
    assert model.n_features_in_ == 2
    assert model.offset_ == -model.threshold_


def test_predict_before_fit_raises():
    x, _, _ = make_data()
    model = IsolationForest()
    with pytest.raises(RuntimeError):
        model.predict(x)
    with pytest.raises(RuntimeError):
        model.decision_function(x)


def test_rejects_bad_contamination():
    for bad in (0, 0.6, -0.1, 1.0):
        with pytest.raises(ValueError):
            IsolationForest(contamination=bad)


def test_rejects_zero_estimators():
    with pytest.raises(ValueError):
        IsolationForest(n_estimators=0)
    with pytest.raises(ValueError):
        IsolationForest(n_estimators=-5)


def test_rejects_nan_input():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2))
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        IsolationForest().fit(x)
    good = rng.normal(size=(50, 2))
    model = IsolationForest(random_state=0).fit(good)
    bad = good.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError):
        model.decision_function(bad)


def test_rejects_1d_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        IsolationForest().fit(rng.normal(size=50))
    model = IsolationForest(random_state=0).fit(rng.normal(size=(50, 2)))
    with pytest.raises(ValueError):
        model.predict(rng.normal(size=2))


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        IsolationForest().fit(np.empty((0, 2)))
    rng = np.random.default_rng(0)
    model = IsolationForest(random_state=0).fit(rng.normal(size=(50, 2)))
    with pytest.raises(ValueError):
        model.predict(np.empty((0, 2)))


def test_score_samples_equals_decision_function():
    x, _, _ = make_data()
    model = IsolationForest(random_state=0).fit(x)
    assert np.allclose(model.score_samples(x), model.decision_function(x))


def test_max_samples_auto():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2))
    model = IsolationForest(max_samples="auto", random_state=0).fit(x)
    assert model.predict(x).shape == (100,)
