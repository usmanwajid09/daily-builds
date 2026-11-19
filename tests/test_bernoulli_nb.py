"""Tests for the Bernoulli naive Bayes classifier."""

import numpy as np
import pytest

from bernoulli_nb.model import BernoulliNB


def test_fits_and_predicts_binary_data():
    rng = np.random.default_rng(0)
    x = (rng.random((100, 4)) < 0.5).astype(int)
    y = (x[:, 0] == 1).astype(int)
    model = BernoulliNB().fit(x, y)
    assert model.score(x, y) >= 0.95


def test_spam_ham_separation():
    rng = np.random.default_rng(1)
    x_spam = (rng.random((50, 3)) < [0.9, 0.9, 0.1]).astype(int)
    x_ham = (rng.random((50, 3)) < [0.1, 0.1, 0.9]).astype(int)
    x = np.vstack([x_spam, x_ham])
    y = np.array([1] * 50 + [0] * 50)
    model = BernoulliNB().fit(x, y)
    assert model.score(x, y) >= 0.95


def test_predict_proba_rows_sum_to_one():
    rng = np.random.default_rng(2)
    x = (rng.random((40, 3)) < 0.5).astype(int)
    y = rng.integers(0, 2, size=40)
    model = BernoulliNB().fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (40, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_matches_argmax_of_proba():
    rng = np.random.default_rng(3)
    x = (rng.random((30, 3)) < 0.5).astype(int)
    y = rng.integers(0, 2, size=30)
    model = BernoulliNB().fit(x, y)
    preds = model.predict(x)
    assert np.array_equal(preds, model.classes_[np.argmax(model.predict_proba(x), axis=1)])


def test_feature_log_prob_matches_counts():
    # Feature 0 is always 1 in class 0 and always 0 in class 1.
    x = np.array([[1, 0], [1, 1], [1, 0], [0, 1], [0, 0], [0, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = BernoulliNB(alpha=1.0).fit(x, y)
    # P(x0=1 | class 0) = (3 + 1) / (3 + 2) = 0.8
    assert np.isclose(np.exp(model.feature_log_prob_[0, 0]), 0.8)
    # P(x0=1 | class 1) = (0 + 1) / (3 + 2) = 0.2
    assert np.isclose(np.exp(model.feature_log_prob_[1, 0]), 0.2)


def test_class_log_prior_matches_frequencies():
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 0, 0, 1])
    model = BernoulliNB().fit(x, y)
    assert np.allclose(model.class_log_prior_, np.log([0.75, 0.25]))


def test_uniform_prior_when_fit_prior_false():
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 0, 0, 1])
    model = BernoulliNB(fit_prior=False).fit(x, y)
    assert np.allclose(model.class_log_prior_, np.log([0.5, 0.5]))


def test_smoothing_avoids_zero_probability():
    x = np.array([[1], [1], [1], [0]])
    y = np.array([0, 0, 0, 1])
    model = BernoulliNB(alpha=1.0).fit(x, y)
    # Feature never seen as 1 in class 1, but smoothing keeps prob > 0.
    assert np.exp(model.feature_log_prob_[1, 0]) > 0


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        BernoulliNB().predict(np.array([[1, 0]]))


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be positive"):
        BernoulliNB(alpha=0.0)


def test_non_binary_features_raise():
    x = np.array([[1, 2], [0, 1], [1, 0]])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="binary features"):
        BernoulliNB().fit(x, y)


def test_non_binary_predict_input_raises():
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 0, 1, 1])
    model = BernoulliNB().fit(x, y)
    with pytest.raises(ValueError, match="binary features"):
        model.predict(np.array([[1, 2]]))


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        BernoulliNB().fit(np.zeros((0, 2), dtype=int), np.array([]))


def test_mismatched_lengths_raises():
    x = np.array([[1, 0], [0, 1]])
    y = np.array([0])
    with pytest.raises(ValueError, match="inconsistent lengths"):
        BernoulliNB().fit(x, y)


def test_single_class_raises():
    x = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="at least 2 classes"):
        BernoulliNB().fit(x, y)


def test_wrong_feature_count_raises():
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 0, 1, 1])
    model = BernoulliNB().fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[1, 0, 1]]))


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2D"):
        BernoulliNB().fit(np.array([1, 0, 1]), np.array([0, 1, 0]))
