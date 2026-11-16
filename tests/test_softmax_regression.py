"""Tests for softmax_regression (multinomial logistic regression)."""

import numpy as np
import pytest

from softmax_regression.model import SoftmaxRegression


def make_blobs(seed=42, n_per_class=100, scale=1.0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.5]])
    X = np.vstack([rng.normal(loc=c, scale=scale, size=(n_per_class, 2))
                   for c in centres])
    y = np.repeat([0, 1, 2], n_per_class)
    return X, y


def split(X, y, seed=42, frac=0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n = int(frac * len(y))
    return X[:n], y[:n], X[n:], y[n:]


def test_separates_three_blobs():
    X, y = make_blobs()
    X_train, y_train, X_test, y_test = split(X, y)
    model = SoftmaxRegression().fit(X_train, y_train)
    acc = np.mean(model.predict(X_test) == y_test)
    assert acc >= 0.95


def test_loss_decreases():
    X, y = make_blobs()
    model = SoftmaxRegression(n_epochs=200).fit(X, y)
    losses = model.loss_history_
    assert len(losses) > 1
    assert losses[-1] < losses[0]
    # Mostly monotonic: at most a couple of small increases.
    increases = sum(b > a for a, b in zip(losses, losses[1:]))
    assert increases <= 2


def test_predict_proba_rows_sum_to_one_and_shape():
    X, y = make_blobs()
    model = SoftmaxRegression(n_epochs=50).fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(y)), atol=1e-8)
    assert np.all((probs >= 0) & (probs <= 1))


def test_predict_before_fit_raises():
    model = SoftmaxRegression()
    X, _ = make_blobs()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(X)


def test_invalid_inputs_raise_value_error():
    X, y = make_blobs()
    model = SoftmaxRegression()
    with pytest.raises(ValueError, match="2D"):
        model.fit(X[0], y)                       # 1D X
    with pytest.raises(ValueError, match="mismatched"):
        model.fit(X, y[:-1])                     # mismatched lengths
    with pytest.raises(ValueError, match="at least 2 classes"):
        model.fit(X, np.zeros(len(y), dtype=int))  # single class
    with pytest.raises(ValueError, match="non-empty"):
        model.fit(np.empty((0, 2)), np.empty(0, dtype=int))
    with pytest.raises(ValueError, match="non-finite"):
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        model.fit(X_bad, y)


def test_classes_contains_all_three_labels():
    X, y = make_blobs()
    model = SoftmaxRegression(n_epochs=50).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1, 2]))
    assert model.coef_.shape == (2, 3)
    assert model.intercept_.shape == (3,)
    assert model.n_epochs_run_ >= 1


def test_works_with_non_zero_based_labels():
    X, y = make_blobs()
    y = y * 10 + 10  # labels become 10, 20, 30
    X_train, y_train, X_test, y_test = split(X, y)
    model = SoftmaxRegression().fit(X_train, y_train)
    np.testing.assert_array_equal(model.classes_, np.array([10, 20, 30]))
    acc = np.mean(model.predict(X_test) == y_test)
    assert acc >= 0.95
    assert set(np.unique(model.predict(X_test))) <= {10, 20, 30}


def test_early_stopping_and_l2():
    X, y = make_blobs()
    model = SoftmaxRegression(n_epochs=1000, l2=0.01, tol=1e-3).fit(X, y)
    # With a loose tolerance the run should stop before the cap.
    assert model.n_epochs_run_ < 1000
    assert len(model.loss_history_) == model.n_epochs_run_
