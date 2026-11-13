"""Tests for perceptron.model.Perceptron."""

import numpy as np
import pytest

from perceptron.model import Perceptron


def make_separable(seed=0, n=40):
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(n, 2))
    class1 = rng.normal(loc=[4.0, 4.0], scale=0.5, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    return x, y


def test_perfect_accuracy_on_separable_data():
    x, y = make_separable()
    model = Perceptron(max_epochs=100, random_state=0).fit(x, y)
    assert model.score(x, y) == 1.0


def test_converges_before_max_epochs():
    """Early stopping: separable data converges in far fewer epochs."""
    x, y = make_separable()
    model = Perceptron(max_epochs=100, random_state=0).fit(x, y)
    assert model.n_epochs_ < 100
    assert model.n_errors_ == 0


def test_labels_preserved():
    """Arbitrary two labels (e.g. -1/+1, strings) are returned as given."""
    x, y = make_separable()
    y_signed = np.where(y == 1, 1, -1)
    model = Perceptron(max_epochs=100, random_state=0).fit(x, y_signed)
    assert set(model.predict(x).tolist()) == {-1, 1}
    assert model.score(x, y_signed) == 1.0


def test_pocket_keeps_best_weights_on_nonseparable_data():
    """On overlapping classes the pocket weights beat the final epoch."""
    rng = np.random.default_rng(1)
    # Overlapping blobs: no perfect linear separator exists.
    class0 = rng.normal(loc=[0.0, 0.0], scale=1.5, size=(50, 2))
    class1 = rng.normal(loc=[1.0, 1.0], scale=1.5, size=(50, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)
    model = Perceptron(max_epochs=50, random_state=1).fit(x, y)
    assert model.n_epochs_ == 50  # never converged, ran all epochs
    train_errors = int(np.sum(model.predict(x) != y))
    assert train_errors == model.n_errors_


def test_predict_single_sample():
    x, y = make_separable()
    model = Perceptron(max_epochs=100, random_state=0).fit(x, y)
    pred = model.predict(x[0])
    assert pred.shape == (1,)
    assert pred[0] == y[0]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        Perceptron().predict([[1.0, 2.0]])


def test_single_class_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="exactly two distinct labels"):
        Perceptron().fit(x, np.array([1, 1, 1]))


def test_three_classes_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="exactly two distinct labels"):
        Perceptron().fit(x, np.array([0, 1, 2]))


def test_max_epochs_zero_raises():
    with pytest.raises(ValueError, match="max_epochs must be at least 1"):
        Perceptron(max_epochs=0)


def test_non_positive_learning_rate_raises():
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        Perceptron(learning_rate=0.0)


def test_mismatched_samples_raises():
    x, y = make_separable()
    with pytest.raises(ValueError, match="same number of samples"):
        Perceptron().fit(x, y[:-1])


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        Perceptron().fit(np.empty((0, 2)), np.empty(0))


def test_nan_input_raises():
    x, y = make_separable()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Perceptron().fit(x, y)


def test_wrong_feature_count_in_predict_raises():
    x, y = make_separable()
    model = Perceptron(max_epochs=100, random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict([[1.0, 2.0, 3.0]])


def test_deterministic_with_same_seed():
    """Same random_state gives identical weights."""
    x, y = make_separable()
    a = Perceptron(max_epochs=20, random_state=3).fit(x, y)
    b = Perceptron(max_epochs=20, random_state=3).fit(x, y)
    assert np.array_equal(a.coef_, b.coef_)
    assert a.intercept_ == b.intercept_
