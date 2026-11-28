"""Tests for the elliptic envelope (robust covariance) anomaly detector."""

import numpy as np
import pytest

from elliptic_envelope.model import EllipticEnvelope


def make_data(seed=0, n_in=200, n_out=10, out_loc=8.0):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n_in, 2))
    outliers = rng.normal(loc=[out_loc, out_loc], scale=1.0, size=(n_out, 2))
    x = np.vstack([inliers, outliers])
    y = np.array([1] * n_in + [-1] * n_out)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_detects_far_outliers():
    x, y = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    preds = model.predict(x)
    recall = float(np.mean(preds[y == -1] == -1))
    assert recall >= 0.8


def test_keeps_most_inliers():
    x, y = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    preds = model.predict(x)
    kept = float(np.mean(preds[y == 1] == 1))
    assert kept >= 0.85


def test_predict_values_are_plus_or_minus_one():
    x, _ = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    preds = model.predict(x)
    assert set(np.unique(preds)) <= {1, -1}


def test_outliers_score_lower_than_inliers():
    x, y = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    scores = model.score_samples(x)
    assert scores[y == -1].mean() < scores[y == 1].mean()


def test_decision_function_sign_matches_predict():
    x, _ = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    decision = model.decision_function(x)
    preds = model.predict(x)
    assert np.array_equal((decision >= 0).astype(int) * 2 - 1, preds)


def test_robust_centre_near_true_centre():
    x, _ = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    assert np.allclose(model.location_, [0.0, 0.0], atol=0.5)


def test_mahalanobis_grows_with_distance():
    x, _ = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    dists = model.mahalanobis(np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]]))
    assert dists[0] < dists[1] < dists[2]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        EllipticEnvelope().predict(np.zeros((2, 2)))


def test_invalid_contamination_raises():
    with pytest.raises(ValueError):
        EllipticEnvelope(contamination=0)
    with pytest.raises(ValueError):
        EllipticEnvelope(contamination=0.6)


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        EllipticEnvelope().fit(np.zeros((2, 5)))


def test_wrong_feature_count_raises():
    x, _ = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=0).fit(x)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, 5)))
