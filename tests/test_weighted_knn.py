"""Tests for the weighted k-NN classifier."""

import numpy as np
import pytest

from weighted_knn.model import WeightedKNNClassifier


def make_data(seed=0, n=60, sep=3.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    x1 = rng.normal(loc=[sep, sep], scale=1.0, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_uniform_weights_high_accuracy():
    x, y = make_data()
    model = WeightedKNNClassifier(n_neighbors=5, weights="uniform").fit(x, y)
    assert model.score(x, y) >= 0.95


def test_distance_weights_high_accuracy():
    x, y = make_data()
    model = WeightedKNNClassifier(n_neighbors=5, weights="distance").fit(x, y)
    assert model.score(x, y) >= 0.95


def test_distance_weighting_beats_uniform_on_overlap():
    # Two heavily overlapping classes: a query point sitting almost on
    # top of a class-0 point should be class 0 even if most neighbours
    # nearby are class 1. Distance weighting must get it right.
    x_train = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.2, 0.0],
        [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1], [4.9, 5.0],
    ])
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    query = np.array([[0.05, 0.05]])
    dist_model = WeightedKNNClassifier(n_neighbors=5, weights="distance").fit(x_train, y_train)
    assert dist_model.predict(query)[0] == 0


def test_predict_proba_rows_sum_to_one():
    x, y = make_data()
    model = WeightedKNNClassifier(n_neighbors=5, weights="distance").fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_manhattan_distance_works():
    x, y = make_data()
    model = WeightedKNNClassifier(n_neighbors=5, weights="distance", p=1).fit(x, y)
    assert model.score(x, y) >= 0.95


def test_multiclass():
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(loc=[i * 4.0, 0.0], scale=1.0, size=(40, 2))
                   for i in range(3)])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    model = WeightedKNNClassifier(n_neighbors=5, weights="distance").fit(x, y)
    assert model.score(x, y) >= 0.95


def test_exact_training_point_predicted_correctly():
    x, y = make_data()
    model = WeightedKNNClassifier(n_neighbors=1, weights="distance").fit(x, y)
    assert np.array_equal(model.predict(x), y)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        WeightedKNNClassifier().predict(np.zeros((2, 2)))


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        WeightedKNNClassifier(n_neighbors=0)
    with pytest.raises(ValueError):
        WeightedKNNClassifier(weights="sqrt")
    with pytest.raises(ValueError):
        WeightedKNNClassifier(p=0)


def test_k_larger_than_samples_raises():
    with pytest.raises(ValueError):
        WeightedKNNClassifier(n_neighbors=10).fit(np.zeros((5, 2)), np.array([0, 1, 0, 1, 0]))


def test_mismatched_xy_raises():
    with pytest.raises(ValueError):
        WeightedKNNClassifier().fit(np.zeros((5, 2)), np.zeros(4))
