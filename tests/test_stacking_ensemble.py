"""Tests for the stacking ensemble classifier."""

import numpy as np
import pytest

from stacking_ensemble.model import (
    DecisionStump,
    SimpleLogisticRegression,
    StackingClassifier,
)


def make_data(seed=0, n=60, sep=3.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    x1 = rng.normal(loc=[sep, sep], scale=1.0, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def make_stacker(**kwargs):
    base = [SimpleLogisticRegression(n_iter=300), DecisionStump()]
    meta = SimpleLogisticRegression(n_iter=300)
    return StackingClassifier(base, meta, n_folds=3, **kwargs)


def test_fits_and_predicts_separable_data():
    x, y = make_data()
    model = make_stacker().fit(x, y)
    assert model.score(x, y) >= 0.95


def test_predict_returns_known_classes():
    x, y = make_data()
    model = make_stacker().fit(x, y)
    preds = model.predict(x)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(y)


def test_predict_proba_rows_sum_to_one():
    x, y = make_data()
    model = make_stacker().fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_use_proba_false_still_works():
    x, y = make_data()
    model = make_stacker(use_proba=False).fit(x, y)
    assert model.score(x, y) >= 0.90


def test_base_learners_refit_on_full_data():
    x, y = make_data()
    model = make_stacker().fit(x, y)
    assert len(model.base_estimators_) == 2
    assert model.meta_estimator_ is not None


def test_predict_before_fit_raises():
    x, _ = make_data()
    with pytest.raises(ValueError, match="not fitted"):
        make_stacker().predict(x)


def test_empty_base_estimators_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        StackingClassifier([], SimpleLogisticRegression())


def test_none_meta_estimator_raises():
    with pytest.raises(ValueError, match="must not be None"):
        StackingClassifier([DecisionStump()], None)


def test_invalid_n_folds_raises():
    with pytest.raises(ValueError, match="n_folds"):
        StackingClassifier([DecisionStump()], SimpleLogisticRegression(), n_folds=1)


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        make_stacker().fit(np.zeros((0, 2)), np.array([]))


def test_mismatched_lengths_raises():
    x, y = make_data()
    with pytest.raises(ValueError, match="inconsistent lengths"):
        make_stacker().fit(x, y[:-1])


def test_single_class_raises():
    x, y = make_data()
    with pytest.raises(ValueError, match="at least 2 classes"):
        make_stacker().fit(x, np.zeros(len(y), dtype=int))


def test_too_few_samples_for_folds_raises():
    x, y = make_data(n=1)
    with pytest.raises(ValueError, match="n_folds"):
        make_stacker().fit(x[:2], y[:2])


def test_1d_x_raises():
    _, y = make_data()
    with pytest.raises(ValueError, match="2D"):
        make_stacker().fit(np.zeros(10), y[:10])


def test_nan_input_raises():
    x, y = make_data()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        make_stacker().fit(x, y)


def test_stump_predicts_majority_on_constant_features():
    x = np.ones((10, 2))
    y = np.array([0] * 7 + [1] * 3)
    stump = DecisionStump().fit(x, y)
    assert set(np.unique(stump.predict(x))) == {0}


def test_logistic_regression_invalid_lr_raises():
    with pytest.raises(ValueError, match="lr must be positive"):
        SimpleLogisticRegression(lr=0.0)
