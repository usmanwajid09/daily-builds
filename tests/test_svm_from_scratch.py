import numpy as np
import pytest

from svm_from_scratch.model import LinearSVC


def _blobs(seed=42, n=60):
    rng = np.random.default_rng(seed)
    x = np.vstack([
        rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2)),
        rng.normal(loc=[4.0, 3.0], scale=1.0, size=(n, 2)),
    ])
    y = np.array([0] * n + [1] * n)
    return x, y


def test_accuracy_on_separable_blobs():
    x, y = _blobs()
    model = LinearSVC(C=1.0, learning_rate=0.01, n_epochs=500,
                      random_state=42).fit(x, y)
    assert float(np.mean(model.predict(x) == y)) >= 0.95


def test_decision_function_sign_matches_predict():
    x, y = _blobs()
    model = LinearSVC(n_epochs=300, random_state=0).fit(x, y)
    scores = model.decision_function(x)
    assert scores.shape == (len(y),)
    preds = np.where(scores >= 0, 1, 0)
    assert np.array_equal(preds, model.predict(x))


def test_coef_shape_matches_features():
    x, y = _blobs()
    model = LinearSVC(n_epochs=10).fit(x, y)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)
    assert list(model.classes_) == [0, 1]


def test_string_labels_work():
    x, y = _blobs()
    y_str = np.where(y == 0, "cat", "dog")
    model = LinearSVC(n_epochs=300, random_state=1).fit(x, y_str)
    assert set(model.predict(x)) <= {"cat", "dog"}
    assert float(np.mean(model.predict(x) == y_str)) >= 0.95


def test_single_point_prediction():
    x, y = _blobs()
    model = LinearSVC(n_epochs=300, random_state=2).fit(x, y)
    assert model.predict(np.array([0.0, 0.0]))[0] == 0
    assert model.predict(np.array([4.0, 3.0]))[0] == 1


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearSVC().predict(np.array([[0.0, 0.0]]))
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearSVC().decision_function(np.array([[0.0, 0.0]]))


def test_one_class_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        LinearSVC().fit(x, np.array([1, 1]))


def test_three_classes_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        LinearSVC().fit(x, np.array([0, 1, 2]))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same number of samples"):
        LinearSVC().fit(np.zeros((3, 2)), np.array([0, 1]))


def test_nan_raises():
    x = np.array([[0.0, np.nan], [1.0, 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        LinearSVC().fit(x, np.array([0, 1]))


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        LinearSVC().fit(np.zeros((0, 2)), np.array([]))


def test_wrong_feature_count_raises():
    x, y = _blobs()
    model = LinearSVC(n_epochs=10).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 3)))


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="C must be"):
        LinearSVC(C=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LinearSVC(learning_rate=-0.1)
    with pytest.raises(ValueError, match="n_epochs"):
        LinearSVC(n_epochs=0)
