"""Tests for the voting ensemble classifier."""

import numpy as np
import pytest

from voting_classifier.model import (
    SimpleKNNClassifier,
    SimpleLogisticRegression,
    VotingClassifier,
)


def make_data(seed=0, n=60, sep=3.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    x1 = rng.normal(loc=[sep, sep], scale=1.0, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def make_voter(voting="soft", **kwargs):
    estimators = [SimpleLogisticRegression(n_iter=300), SimpleKNNClassifier(k=5)]
    return VotingClassifier(estimators, voting=voting, **kwargs)


def test_soft_voting_fits_and_predicts():
    x, y = make_data()
    model = make_voter("soft").fit(x, y)
    assert model.score(x, y) >= 0.95


def test_hard_voting_fits_and_predicts():
    x, y = make_data()
    model = make_voter("hard").fit(x, y)
    assert model.score(x, y) >= 0.95


def test_soft_proba_rows_sum_to_one():
    x, y = make_data()
    model = make_voter("soft").fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_hard_voting_has_no_predict_proba():
    x, y = make_data()
    model = make_voter("hard").fit(x, y)
    with pytest.raises(ValueError, match="only available with soft voting"):
        model.predict_proba(x)


def test_weights_change_hard_vote():
    # Three voters: two say class 1, one says class 0. With a heavy weight
    # on the class-0 voter, the outcome must flip to class 0.
    class Fixed:
        def __init__(self, pred):
            self._pred = pred

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(len(X), self._pred)

        def predict_proba(self, X):
            proba = np.zeros((len(X), 2))
            proba[:, self._pred] = 1.0
            return proba

    x = np.zeros((4, 2))
    y = np.array([0, 0, 1, 1])
    equal = VotingClassifier([Fixed(1), Fixed(1), Fixed(0)], voting="hard").fit(x, y)
    assert list(equal.predict(x)) == [1, 1, 1, 1]
    weighted = VotingClassifier(
        [Fixed(1), Fixed(1), Fixed(0)], voting="hard", weights=[1, 1, 5]
    ).fit(x, y)
    assert list(weighted.predict(x)) == [0, 0, 0, 0]


def test_soft_voting_averages_probabilities():
    class Fixed:
        def __init__(self, proba):
            self._proba = np.asarray(proba, dtype=float)

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.argmax(self._proba, axis=1)

        def predict_proba(self, X):
            return np.tile(self._proba, (len(X), 1))

    x = np.zeros((2, 2))
    y = np.array([0, 1])
    model = VotingClassifier(
        [Fixed([[0.9, 0.1]]), Fixed([[0.4, 0.6]])], voting="soft"
    ).fit(x, y)
    expected = np.array([[0.65, 0.35], [0.65, 0.35]])
    assert np.allclose(model.predict_proba(x), expected)


def test_predict_before_fit_raises():
    x, _ = make_data()
    with pytest.raises(ValueError, match="not fitted"):
        make_voter().predict(x)


def test_empty_estimators_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        VotingClassifier([])


def test_invalid_voting_raises():
    with pytest.raises(ValueError, match="voting must be"):
        VotingClassifier([SimpleKNNClassifier()], voting="medium")


def test_weights_wrong_length_raises():
    with pytest.raises(ValueError, match="weights has length"):
        VotingClassifier([SimpleKNNClassifier(), SimpleKNNClassifier()], weights=[1])


def test_negative_weights_raises():
    with pytest.raises(ValueError, match="must not be negative"):
        VotingClassifier([SimpleKNNClassifier()], weights=[-1])


def test_soft_voting_without_proba_raises():
    class NoProba:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    x, y = make_data()
    with pytest.raises(ValueError, match="predict_proba"):
        VotingClassifier([NoProba()], voting="soft").fit(x, y)


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        make_voter().fit(np.zeros((0, 2)), np.array([]))


def test_mismatched_lengths_raises():
    x, y = make_data()
    with pytest.raises(ValueError, match="inconsistent lengths"):
        make_voter().fit(x, y[:-1])


def test_single_class_raises():
    x, y = make_data()
    with pytest.raises(ValueError, match="at least 2 classes"):
        make_voter().fit(x, np.zeros(len(y), dtype=int))


def test_knn_k_too_large_raises():
    x, y = make_data(n=3)
    with pytest.raises(ValueError, match="larger than the number of samples"):
        SimpleKNNClassifier(k=10).fit(x, y)


def test_knn_k_invalid_raises():
    with pytest.raises(ValueError, match="positive integer"):
        SimpleKNNClassifier(k=0)
