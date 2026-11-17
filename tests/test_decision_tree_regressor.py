"""Tests for the from-scratch CART regression tree."""

import numpy as np
import pytest

from decision_tree_regressor.model import DecisionTreeRegressor


def _leaf_sizes(node):
    """Collect n_samples of every leaf in the tree."""
    if "feature" not in node:
        return [node["n_samples"]]
    return _leaf_sizes(node["left"]) + _leaf_sizes(node["right"])


def test_step_function_fits_almost_exactly():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(200, 1))
    y = np.where(X[:, 0] < 5, 1.0, 10.0)
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)
    assert tree.score(X, y) > 0.99


def test_sine_wave_r2_above_0_8():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 2 * np.pi, size=(150, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, size=150)
    tree = DecisionTreeRegressor(max_depth=5)
    tree.fit(X[:120], y[:120])
    assert tree.score(X[120:], y[120:]) > 0.8


def test_max_depth_1_gives_single_split():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, size=(100, 1))
    y = np.where(X[:, 0] < 5, 1.0, 10.0)
    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(X, y)
    assert tree.n_leaves_ == 2
    assert tree.max_depth_ == 1


def test_min_samples_leaf_respected():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 10, size=(100, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, size=100)
    tree = DecisionTreeRegressor(max_depth=None, min_samples_leaf=10)
    tree.fit(X, y)
    assert all(size >= 10 for size in _leaf_sizes(tree.tree_))


def test_constant_y_predicts_constant():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(50, 2))
    y = np.full(50, 7.5)
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    np.testing.assert_allclose(tree.predict(X), 7.5)
    assert tree.n_leaves_ == 1


def test_predict_shape():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, size=(60, 3))
    y = X[:, 0] + 2 * X[:, 1]
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X[:10])
    assert preds.shape == (10,)


def test_1d_X_raises_value_error():
    tree = DecisionTreeRegressor()
    with pytest.raises(ValueError):
        tree.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_mismatched_lengths_raise_value_error():
    tree = DecisionTreeRegressor()
    with pytest.raises(ValueError):
        tree.fit(np.zeros((5, 2)), np.zeros(4))


def test_empty_input_raises_value_error():
    tree = DecisionTreeRegressor()
    with pytest.raises(ValueError):
        tree.fit(np.zeros((0, 2)), np.zeros(0))


def test_nan_input_raises_value_error():
    tree = DecisionTreeRegressor()
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    with pytest.raises(ValueError):
        tree.fit(X, np.array([1.0, 2.0, 3.0]))


def test_max_depth_0_raises_value_error():
    with pytest.raises(ValueError):
        DecisionTreeRegressor(max_depth=0)


def test_min_samples_leaf_0_raises_value_error():
    with pytest.raises(ValueError):
        DecisionTreeRegressor(min_samples_leaf=0)
