"""Tests for the histogram-based gradient boosting classifier."""

import numpy as np
import pytest

from histogram_gb_classifier.model import HistogramGBClassifier


def make_blobs(seed=0, n=60, sep=3.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    x1 = rng.normal(loc=[sep, sep], scale=1.0, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def make_moons(seed=0, n=120, noise=0.15):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta0 = rng.uniform(0, np.pi, n_half)
    theta1 = rng.uniform(0, np.pi, n_half)
    moon0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    moon1 = np.column_stack([1.0 - np.cos(theta1), 0.5 - np.sin(theta1)])
    x = np.vstack([moon0, moon1]) + rng.normal(scale=noise, size=(n, 2))
    y = np.array([0] * n_half + [1] * n_half)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def _small_model(**kwargs):
    params = dict(n_estimators=30, learning_rate=0.1, max_depth=3,
                  max_bins=8, min_samples_leaf=5)
    params.update(kwargs)
    return HistogramGBClassifier(**params)


def test_fits_blobs_with_high_accuracy():
    x, y = make_blobs()
    model = _small_model().fit(x, y)
    assert model.score(x, y) >= 0.95


def test_fits_nonlinear_moons():
    # Trees must beat a linear model on moons; 0.85 is well above chance.
    x, y = make_moons()
    model = _small_model(n_estimators=60).fit(x, y)
    assert model.score(x, y) >= 0.85


def test_predict_proba_rows_sum_to_one():
    x, y = make_blobs()
    model = _small_model().fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_multiclass_one_vs_rest():
    rng = np.random.default_rng(4)
    x = np.vstack([rng.normal(loc=[i * 4.0, 0.0], scale=1.0, size=(40, 2))
                   for i in range(3)])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    model = _small_model().fit(x, y)
    assert model.score(x, y) >= 0.90
    assert len(model.boosters_) == 3


def test_more_estimators_do_not_hurt_training_fit():
    x, y = make_blobs()
    short = _small_model(n_estimators=5).fit(x, y)
    long = _small_model(n_estimators=40).fit(x, y)
    assert long.score(x, y) >= short.score(x, y) - 0.05


def test_constant_feature_does_not_crash():
    x, y = make_blobs()
    x = np.column_stack([x, np.ones(len(y))])
    model = _small_model().fit(x, y)
    assert model.score(x, y) >= 0.90


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        HistogramGBClassifier().predict(np.zeros((2, 2)))


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        HistogramGBClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        HistogramGBClassifier(learning_rate=0)
    with pytest.raises(ValueError):
        HistogramGBClassifier(max_depth=0)
    with pytest.raises(ValueError):
        HistogramGBClassifier(max_bins=1)


def test_mismatched_xy_raises():
    with pytest.raises(ValueError):
        HistogramGBClassifier().fit(np.zeros((5, 2)), np.zeros(4))


def test_single_class_raises():
    with pytest.raises(ValueError):
        HistogramGBClassifier().fit(np.zeros((5, 2)), np.zeros(5, dtype=int))
