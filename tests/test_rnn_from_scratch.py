"""Tests for rnn_from_scratch: vanilla RNN sequence classifier (numpy only)."""
import numpy as np
import pytest

from rnn_from_scratch import RNNClassifier
from rnn_from_scratch.model import make_parity_dataset


def _parity(n=200, seq_len=6, seed=0):
    return make_parity_dataset(n, seq_len=seq_len, seed=seed)


def test_loss_decreases_over_training():
    X, y = _parity(n=200, seed=0)
    clf = RNNClassifier(hidden_size=32, epochs=30, seed=42)
    clf.fit(X, y)
    assert len(clf.loss_history_) == 30
    assert clf.loss_history_[-1] < clf.loss_history_[0]
    assert clf.loss_history_[-1] < 0.69  # below chance-level loss ln(2)


def test_predict_returns_valid_labels_with_correct_shape():
    X, y = _parity(n=100, seed=1)
    clf = RNNClassifier(hidden_size=16, epochs=20, seed=42).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_proba_rows_sum_to_one():
    X, y = _parity(n=50, seed=2)
    clf = RNNClassifier(hidden_size=16, epochs=10, seed=42).fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (50, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_score_in_zero_to_one():
    X, y = _parity(n=100, seed=3)
    clf = RNNClassifier(hidden_size=16, epochs=10, seed=42).fit(X, y)
    score = clf.score(X, y)
    assert 0.0 <= score <= 1.0


def test_accuracy_above_chance_after_training():
    X_train, y_train = _parity(n=300, seed=0)
    X_test, y_test = _parity(n=100, seed=99)
    clf = RNNClassifier(hidden_size=32, epochs=150, seed=42).fit(X_train, y_train)
    assert clf.score(X_test, y_test) > 0.5
    assert clf.score(X_train, y_train) > 0.8


def test_hidden_size_validation():
    with pytest.raises(ValueError, match="hidden_size"):
        RNNClassifier(hidden_size=0)
    with pytest.raises(ValueError, match="hidden_size"):
        RNNClassifier(hidden_size=-4)


def test_learning_rate_validation():
    with pytest.raises(ValueError, match="learning_rate"):
        RNNClassifier(learning_rate=0)
    with pytest.raises(ValueError, match="learning_rate"):
        RNNClassifier(learning_rate=-0.1)


def test_epochs_validation():
    with pytest.raises(ValueError, match="epochs"):
        RNNClassifier(epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        RNNClassifier(epochs=-5)


def test_X_must_be_3d():
    X_2d = np.zeros((10, 6))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="3D"):
        RNNClassifier(epochs=1).fit(X_2d, y)


def test_y_length_must_match_X():
    X, y = _parity(n=20, seed=4)
    with pytest.raises(ValueError, match="labels"):
        RNNClassifier(epochs=1).fit(X, y[:10])


def test_single_class_raises():
    X, _ = _parity(n=20, seed=5)
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="at least 2 classes"):
        RNNClassifier(epochs=1).fit(X, y)


def test_fit_returns_self():
    X, y = _parity(n=20, seed=6)
    clf = RNNClassifier(hidden_size=8, epochs=2, seed=42)
    assert clf.fit(X, y) is clf


def test_deterministic_with_same_seed():
    X, y = _parity(n=100, seed=7)
    clf1 = RNNClassifier(hidden_size=16, epochs=20, seed=42).fit(X, y)
    clf2 = RNNClassifier(hidden_size=16, epochs=20, seed=42).fit(X, y)
    assert np.array_equal(clf1.predict(X), clf2.predict(X))
    assert clf1.loss_history_ == clf2.loss_history_


def test_loss_history_length_equals_epochs():
    X, y = _parity(n=50, seed=8)
    clf = RNNClassifier(hidden_size=8, epochs=7, seed=42).fit(X, y)
    assert len(clf.loss_history_) == 7
    assert all(np.isfinite(loss) for loss in clf.loss_history_)
