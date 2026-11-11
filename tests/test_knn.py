import numpy as np
import pytest

from knn_from_scratch.model import KNNClassifier


def _toy():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return x, y


def test_predicts_nearby_class():
    x, y = _toy()
    model = KNNClassifier(k=3).fit(x, y)
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[5.1, 5.1]]))[0] == 1


def test_k1_returns_nearest_label():
    x, y = _toy()
    model = KNNClassifier(k=1).fit(x, y)
    assert model.predict(np.array([[0.9, 0.0]]))[0] == 0
    assert model.predict(np.array([[5.9, 5.0]]))[0] == 1


def test_batch_prediction_shape():
    x, y = _toy()
    model = KNNClassifier(k=3).fit(x, y)
    preds = model.predict(np.array([[0.0, 0.0], [5.0, 5.0], [0.5, 0.5]]))
    assert preds.shape == (3,)
    assert set(preds.tolist()) <= {0, 1}


def test_tie_break_is_deterministic():
    # Point exactly between a 0 and a 1, k=2: vote ties 1-1, nearest wins.
    x = np.array([[0.0, 0.0], [2.0, 0.0]])
    y = np.array([0, 1])
    model = KNNClassifier(k=2).fit(x, y)
    first = model.predict(np.array([[0.9, 0.0]]))[0]
    second = model.predict(np.array([[0.9, 0.0]]))[0]
    assert first == second == 0


def test_rejects_bad_k():
    with pytest.raises(ValueError):
        KNNClassifier(k=0)
    with pytest.raises(ValueError):
        KNNClassifier(k=-3)


def test_rejects_k_larger_than_training_set():
    x, y = _toy()
    with pytest.raises(ValueError):
        KNNClassifier(k=7).fit(x, y)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        KNNClassifier().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0]))


def test_rejects_empty_data():
    with pytest.raises(ValueError):
        KNNClassifier().fit(np.empty((0, 2)), np.array([]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNClassifier().predict(np.array([[0.0, 0.0]]))


def test_rejects_wrong_feature_count():
    x, y = _toy()
    model = KNNClassifier(k=3).fit(x, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0, 0.0]]))
