import numpy as np
import pytest

from random_forest.model import RandomForestClassifier


def _blobs(seed=42, n=60):
    rng = np.random.default_rng(seed)
    centres = [[0.0, 0.0], [4.0, 3.0], [0.0, 5.0]]
    xs, ys = [], []
    for label, centre in enumerate(centres):
        xs.append(rng.normal(loc=centre, scale=0.8, size=(n, 2)))
        ys.extend([label] * n)
    return np.vstack(xs), np.array(ys)


def test_accuracy_on_three_blobs():
    x, y = _blobs()
    model = RandomForestClassifier(n_estimators=50, max_depth=6,
                                   random_state=42).fit(x, y)
    assert float(np.mean(model.predict(x) == y)) >= 0.95


def test_predict_proba_sums_to_one():
    x, y = _blobs()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    proba = model.predict_proba(x[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_matches_highest_proba():
    x, y = _blobs()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    proba = model.predict_proba(x)
    assert np.array_equal(model.predict(x), model.classes_[np.argmax(proba, axis=1)])


def test_single_tree_works():
    x, y = _blobs()
    model = RandomForestClassifier(n_estimators=1, random_state=7).fit(x, y)
    assert float(np.mean(model.predict(x) == y)) >= 0.9


def test_feature_importances_sum_to_one():
    x, y = _blobs()
    model = RandomForestClassifier(n_estimators=20, random_state=3).fit(x, y)
    assert model.feature_importances_.shape == (2,)
    assert np.isclose(model.feature_importances_.sum(), 1.0)
    assert np.all(model.feature_importances_ >= 0)


def test_max_features_int():
    x, y = _blobs()
    model = RandomForestClassifier(n_estimators=10, max_features=1,
                                   random_state=5).fit(x, y)
    assert float(np.mean(model.predict(x) == y)) >= 0.9


def test_max_features_all():
    x, y = _blobs()
    model = RandomForestClassifier(n_estimators=10, max_features=None,
                                   random_state=5).fit(x, y)
    assert float(np.mean(model.predict(x) == y)) >= 0.9


def test_string_labels_work():
    x, y = _blobs()
    y_str = np.array(["a", "b", "c"])[y]
    model = RandomForestClassifier(n_estimators=10, random_state=1).fit(x, y_str)
    assert set(model.predict(x)) <= {"a", "b", "c"}


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RandomForestClassifier().predict(np.array([[0.0, 0.0]]))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same number of samples"):
        RandomForestClassifier().fit(np.zeros((3, 2)), np.array([0, 1]))


def test_nan_raises():
    x = np.array([[0.0, np.nan], [1.0, 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        RandomForestClassifier().fit(x, np.array([0, 1]))


def test_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        RandomForestClassifier().fit(np.zeros((0, 2)), np.array([]))


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_estimators"):
        RandomForestClassifier(n_estimators=0)
    with pytest.raises(ValueError, match="max_features"):
        RandomForestClassifier(max_features=0)
    with pytest.raises(ValueError, match="max_depth"):
        RandomForestClassifier(max_depth=0)
    with pytest.raises(ValueError, match="max_features"):
        RandomForestClassifier(max_features=5).fit(*_blobs())
