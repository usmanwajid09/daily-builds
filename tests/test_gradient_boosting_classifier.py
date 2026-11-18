"""Tests for gradient_boosting_classifier.model.GradientBoostingClassifier."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from gradient_boosting_classifier.model import GradientBoostingClassifier


def _blobs(seed=0, n=120):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n // 2, 2))
    c1 = rng.normal(loc=[4.0, 3.0], scale=1.0, size=(n // 2, 2))
    x = np.vstack([c0, c1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    perm = rng.permutation(n)
    return x[perm], y[perm]


def _three_blobs(seed=0, n=150):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k, loc in enumerate([(0, 0), (5, 0), (2.5, 4)]):
        xs.append(rng.normal(loc=loc, scale=0.8, size=(n // 3, 2)))
        ys.extend([k] * (n // 3))
    x = np.vstack(xs)
    y = np.array(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_accuracy_on_separable_blobs():
    x, y = _blobs()
    model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1,
                                       max_depth=2, random_state=0).fit(x, y)
    assert model.score(x, y) >= 0.98


def test_predict_proba_rows_sum_to_one():
    x, y = _blobs()
    model = GradientBoostingClassifier(n_estimators=10, random_state=0)
    model.fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_multiclass_accuracy():
    x, y = _three_blobs()
    model = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1,
                                       max_depth=2, random_state=0).fit(x, y)
    assert model.score(x, y) >= 0.95
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_staged_predict_improves_or_matches():
    x, y = _blobs()
    model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1,
                                       max_depth=2, random_state=0).fit(x, y)
    staged = list(model.staged_predict(x))
    assert len(staged) == 20
    first = float(np.mean(staged[0] == y))
    last = float(np.mean(staged[-1] == y))
    assert last >= first
    assert np.array_equal(staged[-1], model.predict(x))


def test_feature_importances_sum_to_one():
    x, y = _blobs()
    model = GradientBoostingClassifier(n_estimators=20, random_state=0)
    model.fit(x, y)
    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_.sum() == pytest.approx(1.0)
    assert np.all(model.feature_importances_ >= 0)


def test_predict_before_fit_raises():
    x, _ = _blobs()
    model = GradientBoostingClassifier()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(x)


def test_single_class_raises():
    x = np.random.default_rng(0).normal(size=(20, 2))
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="at least 2 classes"):
        GradientBoostingClassifier().fit(x, y)


def test_mismatched_lengths_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="different lengths"):
        GradientBoostingClassifier().fit(x, y[:-1])


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2-dimensional"):
        GradientBoostingClassifier().fit(np.zeros(10), np.zeros(10, dtype=int))


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_estimators"):
        GradientBoostingClassifier(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        GradientBoostingClassifier(learning_rate=0.0)
    with pytest.raises(ValueError, match="max_depth"):
        GradientBoostingClassifier(max_depth=0)
    with pytest.raises(ValueError, match="subsample"):
        GradientBoostingClassifier(subsample=1.5)


def test_classes_attribute():
    x, y = _three_blobs()
    model = GradientBoostingClassifier(n_estimators=5, random_state=0)
    model.fit(x, y)
    assert list(model.classes_) == [0, 1, 2]
    assert model.n_classes_ == 3


def test_reproducible_with_random_state():
    x, y = _blobs()
    kwargs = dict(n_estimators=10, learning_rate=0.1, max_depth=2,
                  subsample=0.8, random_state=7)
    a = GradientBoostingClassifier(**kwargs).fit(x, y)
    b = GradientBoostingClassifier(**kwargs).fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))
