import numpy as np
import pytest

from linreg_gd.model import LinearRegressionGD


def test_recovers_known_line_without_noise():
    x = np.linspace(0, 10, 50)
    y = 3.0 * x + 2.0
    model = LinearRegressionGD(learning_rate=0.01, n_iters=2000).fit(x, y)
    assert model.w_ == pytest.approx(3.0, abs=0.05)
    assert model.b_ == pytest.approx(2.0, abs=0.05)


def test_predict_matches_fitted_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1
    model = LinearRegressionGD(learning_rate=0.05, n_iters=3000).fit(x, y)
    preds = model.predict(np.array([10.0]))
    assert preds[0] == pytest.approx(21.0, abs=0.5)


def test_rejects_mismatched_lengths():
    model = LinearRegressionGD()
    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]), np.array([1, 2]))


def test_rejects_empty_data():
    model = LinearRegressionGD()
    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))


def test_predict_before_fit_raises():
    model = LinearRegressionGD()
    with pytest.raises(RuntimeError):
        model.predict(np.array([1.0, 2.0]))


def test_invalid_hyperparameters_rejected():
    with pytest.raises(ValueError):
        LinearRegressionGD(learning_rate=0)
    with pytest.raises(ValueError):
        LinearRegressionGD(n_iters=0)
