import numpy as np
import pytest

from decision_tree.model import DecisionTreeClassifier


def _toy():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return x, y


def test_predicts_nearby_class():
    x, y = _toy()
    model = DecisionTreeClassifier().fit(x, y)
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[5.1, 5.1]]))[0] == 1


def test_pure_leaf_when_one_class():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([1, 1, 1])
    model = DecisionTreeClassifier().fit(x, y)
    assert model.depth == 0
    assert model.predict(np.array([[9.0, 9.0]]))[0] == 1


def test_max_depth_is_respected():
    x, y = _toy()
    model = DecisionTreeClassifier(max_depth=1).fit(x, y)
    assert model.depth == 1


def test_predict_proba_sums_to_one():
    x, y = _toy()
    model = DecisionTreeClassifier().fit(x, y)
    proba = model.predict_proba(np.array([[0.1, 0.1], [5.1, 5.1]]))
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 0] > proba[0, 1]  # point near class 0
    assert proba[1, 1] > proba[1, 0]  # point near class 1


def test_three_classes():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [2.5, 5.0], [2.5, 6.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    model = DecisionTreeClassifier().fit(x, y)
    preds = model.predict(x)
    assert np.mean(preds == y) == 1.0


def test_rejects_bad_max_depth():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=0)
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=-1)


def test_rejects_bad_min_samples_split():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0]))


def test_rejects_empty_data():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.empty((0, 2)), np.array([]))


def test_rejects_nan_values():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.array([[0.0, np.nan]]), np.array([0]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict(np.array([[0.0, 0.0]]))


def test_rejects_wrong_feature_count():
    x, y = _toy()
    model = DecisionTreeClassifier().fit(x, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0, 0.0]]))
