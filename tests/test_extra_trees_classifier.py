"""Tests for extra_trees_classifier.model.ExtraTreesClassifier."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from extra_trees_classifier.model import ExtraTreesClassifier


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
    model = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(x, y)
    assert model.score(x, y) >= 0.98


def test_predict_proba_rows_sum_to_one():
    x, y = _blobs()
    model = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_multiclass_accuracy():
    x, y = _three_blobs()
    model = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(x, y)
    assert model.score(x, y) >= 0.95
    assert model.predict_proba(x).shape == (len(y), 3)


def test_forest_beats_single_tree_on_noisy_data():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=(n, 2))
    y = ((x[:, 0] + x[:, 1] > 0) & (rng.random(n) > 0.15)).astype(int)
    forest = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(x, y)
    single = ExtraTreesClassifier(n_estimators=1, random_state=0).fit(x, y)
    assert forest.score(x, y) >= single.score(x, y) - 0.05


def test_feature_importances_sum_to_one():
    x, y = _blobs()
    model = ExtraTreesClassifier(n_estimators=20, random_state=0).fit(x, y)
    assert model.feature_importances_.shape == (2,)
    assert model.feature_importances_.sum() == pytest.approx(1.0)
    assert np.all(model.feature_importances_ >= 0)


def test_predict_before_fit_raises():
    x, _ = _blobs()
    with pytest.raises(ValueError, match="not fitted"):
        ExtraTreesClassifier().predict(x)


def test_single_class_raises():
    x = np.random.default_rng(0).normal(size=(20, 2))
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="at least 2 classes"):
        ExtraTreesClassifier().fit(x, y)


def test_mismatched_lengths_raises():
    x, y = _blobs()
    with pytest.raises(ValueError, match="different lengths"):
        ExtraTreesClassifier().fit(x, y[:-1])


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2-dimensional"):
        ExtraTreesClassifier().fit(np.zeros(10), np.zeros(10, dtype=int))


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_estimators"):
        ExtraTreesClassifier(n_estimators=0)
    with pytest.raises(ValueError, match="max_depth"):
        ExtraTreesClassifier(max_depth=0)
    with pytest.raises(ValueError, match="min_samples_split"):
        ExtraTreesClassifier(min_samples_split=1)
    with pytest.raises(ValueError, match="min_samples_leaf"):
        ExtraTreesClassifier(min_samples_leaf=0)
    with pytest.raises(ValueError, match="max_features"):
        ExtraTreesClassifier(max_features="log2")


def test_reproducible_with_random_state():
    x, y = _blobs()
    a = ExtraTreesClassifier(n_estimators=20, random_state=7).fit(x, y)
    b = ExtraTreesClassifier(n_estimators=20, random_state=7).fit(x, y)
    assert np.array_equal(a.predict(x), b.predict(x))
    assert np.allclose(a.predict_proba(x), b.predict_proba(x))


def test_bootstrap_mode_runs():
    x, y = _blobs()
    model = ExtraTreesClassifier(n_estimators=20, bootstrap=True,
                                 random_state=0).fit(x, y)
    assert model.score(x, y) >= 0.95
    assert len(model.trees_) == 20
