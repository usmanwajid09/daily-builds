import numpy as np
import pytest

from naive_bayes.model import GaussianNB


def _toy():
    rng = np.random.default_rng(0)
    n = 40
    class0 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(n, 2))
    class1 = rng.normal(loc=[5.0, 5.0], scale=0.5, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    return x, y


def test_separates_gaussian_classes():
    x, y = _toy()
    model = GaussianNB().fit(x, y)
    preds = model.predict(x)
    assert np.mean(preds == y) == 1.0


def test_predicts_near_class_centre():
    x, y = _toy()
    model = GaussianNB().fit(x, y)
    assert model.predict(np.array([[0.1, -0.1]]))[0] == 0
    assert model.predict(np.array([[5.1, 4.9]]))[0] == 1


def test_predict_proba_sums_to_one_and_is_confident():
    x, y = _toy()
    model = GaussianNB().fit(x, y)
    proba = model.predict_proba(np.array([[0.0, 0.0], [5.0, 5.0]]))
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 0] > 0.99
    assert proba[1, 1] > 0.99


def test_handles_constant_feature():
    # Feature 0 is identical inside each class; variance floor must save it.
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.2],
                  [2.0, 5.0], [2.0, 6.0], [2.0, 5.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = GaussianNB().fit(x, y)
    assert model.predict(np.array([[1.0, 0.5]]))[0] == 0
    assert model.predict(np.array([[2.0, 5.2]]))[0] == 1


def test_handles_unequal_class_sizes():
    rng = np.random.default_rng(1)
    class0 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(10, 2))
    class1 = rng.normal(loc=[5.0, 5.0], scale=0.5, size=(50, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * 10 + [1] * 50)
    model = GaussianNB().fit(x, y)
    assert np.mean(model.predict(x) == y) == 1.0


def test_rejects_negative_var_smoothing():
    with pytest.raises(ValueError):
        GaussianNB(var_smoothing=-1e-9)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        GaussianNB().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0]))


def test_rejects_empty_data():
    with pytest.raises(ValueError):
        GaussianNB().fit(np.empty((0, 2)), np.array([]))


def test_rejects_nan_values():
    with pytest.raises(ValueError):
        GaussianNB().fit(np.array([[0.0, np.nan]]), np.array([0]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GaussianNB().predict(np.array([[0.0, 0.0]]))


def test_rejects_wrong_feature_count():
    x, y = _toy()
    model = GaussianNB().fit(x, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0, 0.0]]))
