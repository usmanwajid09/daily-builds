"""Tests for the Adaline classifier."""

import numpy as np
import pytest

from adaline_classifier.model import AdalineClassifier


def make_data(seed=0, n=60, sep=3.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    x1 = rng.normal(loc=[sep, sep], scale=1.0, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_fits_and_predicts_with_high_accuracy():
    x, y = make_data()
    model = AdalineClassifier(learning_rate=0.05, n_iter=500).fit(x, y)
    assert model.score(x, y) >= 0.95


def test_predict_returns_known_labels():
    x, y = make_data()
    model = AdalineClassifier(learning_rate=0.05, n_iter=500).fit(x, y)
    preds = model.predict(x)
    assert set(np.unique(preds)) <= {0, 1}
    assert preds.shape == y.shape


def test_predict_proba_rows_sum_to_one():
    x, y = make_data()
    model = AdalineClassifier(learning_rate=0.05, n_iter=500).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_multiclass_one_vs_rest():
    rng = np.random.default_rng(1)
    centres = [[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]]  # triangle: each OvR split is separable
    x = np.vstack([rng.normal(loc=c, scale=1.0, size=(40, 2)) for c in centres])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    model = AdalineClassifier(learning_rate=0.05, n_iter=500).fit(x, y)
    assert model.score(x, y) >= 0.90
    assert model.coef_.shape == (3, 2)


def test_standardisation_handles_unscaled_features():
    # Features on wildly different scales: without standardisation
    # Adaline would struggle; with it, accuracy stays high.
    rng = np.random.default_rng(2)
    x0 = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1000.0], size=(60, 2))
    x1 = rng.normal(loc=[3.0, 3000.0], scale=[1.0, 1000.0], size=(60, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 60 + [1] * 60)
    model = AdalineClassifier(learning_rate=0.05, n_iter=500).fit(x, y)
    assert model.score(x, y) >= 0.90


def test_cost_decreases_over_training():
    x, y = make_data()
    model = AdalineClassifier(learning_rate=0.05, n_iter=500).fit(x, y)
    costs = model.costs_[0]
    assert costs[-1] < costs[0]


def test_predict_before_fit_raises():
    model = AdalineClassifier()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 2)))


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        AdalineClassifier(learning_rate=0)
    with pytest.raises(ValueError):
        AdalineClassifier(n_iter=0)


def test_mismatched_xy_raises():
    with pytest.raises(ValueError):
        AdalineClassifier().fit(np.zeros((5, 2)), np.zeros(4))


def test_single_class_raises():
    with pytest.raises(ValueError):
        AdalineClassifier().fit(np.zeros((5, 2)), np.zeros(5, dtype=int))


def test_wrong_feature_count_raises():
    x, y = make_data()
    model = AdalineClassifier(learning_rate=0.05, n_iter=100).fit(x, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, 5)))
