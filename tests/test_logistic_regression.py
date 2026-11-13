import numpy as np
import pytest

from logistic_regression.model import LogisticRegression


def _toy():
    rng = np.random.default_rng(0)
    n = 50
    class0 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(n, 2))
    class1 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    return x, y


def test_separates_linearly_separable_data():
    x, y = _toy()
    model = LogisticRegression(n_epochs=500).fit(x, y)
    preds = model.predict(x)
    assert np.mean(preds == y) >= 0.99


def test_loss_decreases_during_training():
    x, y = _toy()
    model = LogisticRegression(n_epochs=200).fit(x, y)
    history = model.loss_history_
    assert len(history) > 1
    assert history[-1] < history[0]


def test_predict_proba_sums_to_one_and_is_confident():
    x, y = _toy()
    model = LogisticRegression(n_epochs=500).fit(x, y)
    proba = model.predict_proba(np.array([[-2.0, -2.0], [2.0, 2.0]]))
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 0] > 0.9  # clearly class 0
    assert proba[1, 1] > 0.9  # clearly class 1


def test_l2_regularisation_shrinks_weights():
    x, y = _toy()
    plain = LogisticRegression(n_epochs=500, l2=0.0).fit(x, y)
    shrunk = LogisticRegression(n_epochs=500, l2=100.0).fit(x, y)
    assert np.linalg.norm(shrunk.coef_) < np.linalg.norm(plain.coef_)


def test_single_sample_prediction():
    x, y = _toy()
    model = LogisticRegression(n_epochs=300).fit(x, y)
    assert model.predict(np.array([-2.0, -2.0])).shape == (1,)
    assert model.predict(np.array([2.0, 2.0]))[0] == 1


def test_rejects_bad_learning_rate():
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=0)
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=-0.1)


def test_rejects_bad_n_epochs():
    with pytest.raises(ValueError):
        LogisticRegression(n_epochs=0)


def test_rejects_negative_l2():
    with pytest.raises(ValueError):
        LogisticRegression(l2=-1.0)


def test_rejects_single_class():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        LogisticRegression().fit(x, np.array([1, 1]))


def test_rejects_three_classes():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError):
        LogisticRegression().fit(x, np.array([0, 1, 2]))


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0]))


def test_rejects_empty_data():
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.empty((0, 2)), np.array([]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LogisticRegression().predict(np.array([[0.0, 0.0]]))


def test_rejects_wrong_feature_count():
    x, y = _toy()
    model = LogisticRegression(n_epochs=10).fit(x, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0, 0.0]]))
