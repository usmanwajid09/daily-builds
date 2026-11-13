"""Tests for adaboost_stumps.model.AdaBoostStumps and DecisionStump."""

import numpy as np
import pytest

from adaboost_stumps.model import AdaBoostStumps, DecisionStump


def make_linearly_separable(seed=0, n=40):
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(n, 2))
    class1 = rng.normal(loc=[4.0, 4.0], scale=0.5, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    return x, y


def make_diagonal(seed=0, n=100):
    """Gaussian points, label 1 where x0 + x1 > 0.5 (diagonal boundary)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] + x[:, 1] > 0.5).astype(int)
    return x, y


def test_high_accuracy_on_separable_data():
    x, y = make_linearly_separable()
    model = AdaBoostStumps(n_estimators=10).fit(x, y)
    assert model.score(x, y) == 1.0


def test_beats_single_stump_on_diagonal():
    """Boosting combines axis stumps to approximate a diagonal boundary.

    A single stump manages only ~70-80% on this data; boosting pushes
    it to ~95%+ by combining stumps on both axes.
    """
    x, y = make_diagonal()
    single = AdaBoostStumps(n_estimators=1).fit(x, y)
    boosted = AdaBoostStumps(n_estimators=30).fit(x, y)
    assert boosted.score(x, y) > single.score(x, y)
    assert boosted.score(x, y) >= 0.9


def test_staged_predict_matches_predict():
    x, y = make_diagonal()
    model = AdaBoostStumps(n_estimators=10).fit(x, y)
    stages = model.staged_predict(x)
    assert len(stages) == model.n_estimators_
    assert np.array_equal(stages[-1], model.predict(x))


def test_staged_accuracy_improves_overall():
    """Later stages are at least as good as the first on training data."""
    x, y = make_diagonal()
    model = AdaBoostStumps(n_estimators=20).fit(x, y)
    stages = model.staged_predict(x)
    accs = [float(np.mean(s == y)) for s in stages]
    assert accs[-1] >= accs[0]


def test_feature_importances_sum_to_one():
    x, y = make_diagonal()
    model = AdaBoostStumps(n_estimators=20).fit(x, y)
    assert model.feature_importances_.shape == (2,)
    assert np.isclose(model.feature_importances_.sum(), 1.0)
    assert (model.feature_importances_ >= 0).all()


def test_stump_weights_positive():
    x, y = make_linearly_separable()
    model = AdaBoostStumps(n_estimators=10).fit(x, y)
    assert (model.stump_weights_ > 0).all()


def test_early_stop_on_perfect_stump():
    """Perfectly separable by one stump: stops after 1 estimator."""
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = AdaBoostStumps(n_estimators=50).fit(x, y)
    assert model.n_estimators_ == 1
    assert model.score(x, y) == 1.0


def test_labels_preserved():
    x, y = make_linearly_separable()
    y_signed = np.where(y == 1, 1, -1)
    model = AdaBoostStumps(n_estimators=10).fit(x, y_signed)
    assert set(model.predict(x).tolist()) == {-1, 1}
    assert model.score(x, y_signed) == 1.0


def test_predict_single_sample():
    x, y = make_linearly_separable()
    model = AdaBoostStumps(n_estimators=10).fit(x, y)
    pred = model.predict(x[0])
    assert pred.shape == (1,)
    assert pred[0] == y[0]


def test_stump_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        DecisionStump().predict([[1.0, 2.0]])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        AdaBoostStumps().predict([[1.0, 2.0]])


def test_staged_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        AdaBoostStumps().staged_predict([[1.0, 2.0]])


def test_single_class_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="exactly two distinct labels"):
        AdaBoostStumps().fit(x, np.array([1, 1]))


def test_three_classes_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="exactly two distinct labels"):
        AdaBoostStumps().fit(x, np.array([0, 1, 2]))


def test_n_estimators_zero_raises():
    with pytest.raises(ValueError, match="at least 1"):
        AdaBoostStumps(n_estimators=0)


def test_non_positive_learning_rate_raises():
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        AdaBoostStumps(learning_rate=0.0)


def test_mismatched_samples_raises():
    x, y = make_linearly_separable()
    with pytest.raises(ValueError, match="same number of samples"):
        AdaBoostStumps().fit(x, y[:-1])


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        AdaBoostStumps().fit(np.empty((0, 2)), np.empty(0))


def test_nan_input_raises():
    x, y = make_linearly_separable()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        AdaBoostStumps().fit(x, y)


def test_wrong_feature_count_in_predict_raises():
    x, y = make_linearly_separable()
    model = AdaBoostStumps(n_estimators=10).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict([[1.0, 2.0, 3.0]])


def test_all_constant_features_raise():
    x = np.ones((6, 2))
    y = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="could not train any stump"):
        AdaBoostStumps().fit(x, y)


def test_deterministic():
    """Same data gives identical stump weights every time."""
    x, y = make_diagonal()
    a = AdaBoostStumps(n_estimators=15).fit(x, y)
    b = AdaBoostStumps(n_estimators=15).fit(x, y)
    assert np.array_equal(a.stump_weights_, b.stump_weights_)
    assert np.array_equal(a.predict(x), b.predict(x))
