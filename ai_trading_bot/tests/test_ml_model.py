import numpy as np
import pytest

from ai_trading_bot.ml_model import LogisticRegressionGD, StandardScaler, _sigmoid


def test_sigmoid_matches_definition_and_is_stable():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = _sigmoid(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[2], 0.5)
    assert out[0] < 1e-6
    assert out[-1] > 1.0 - 1e-6
    # matches naive formula away from the overflow-prone extremes
    naive = 1.0 / (1.0 + np.exp(-z[1:4]))
    np.testing.assert_allclose(out[1:4], naive, rtol=1e-9)


def test_standard_scaler_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(500, 3))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Xs.std(axis=0), 1.0, atol=1e-9)


def test_standard_scaler_constant_feature_no_divide_by_zero():
    X = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    assert np.all(np.isfinite(Xs))
    np.testing.assert_allclose(Xs[:, 0], 0.0)  # constant feature -> all zeros, not nan/inf


def test_standard_scaler_transform_before_fit_raises():
    scaler = StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((5, 2)))


def test_logistic_regression_learns_linearly_separable_data():
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(0, 1, size=(n, 2))
    true_w = np.array([3.0, -2.0])
    y = (X @ true_w > 0).astype(float)
    model = LogisticRegressionGD(learning_rate=0.5, n_iters=1000, l2=0.0).fit(X, y)
    preds = model.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.95


def test_logistic_regression_rejects_bad_params():
    with pytest.raises(ValueError):
        LogisticRegressionGD(learning_rate=0)
    with pytest.raises(ValueError):
        LogisticRegressionGD(n_iters=0)
    with pytest.raises(ValueError):
        LogisticRegressionGD(l2=-1)


def test_logistic_regression_rejects_mismatched_lengths():
    model = LogisticRegressionGD()
    with pytest.raises(ValueError):
        model.fit(np.zeros((5, 2)), np.zeros(3))


def test_logistic_regression_rejects_non_binary_labels():
    model = LogisticRegressionGD()
    with pytest.raises(ValueError):
        model.fit(np.zeros((3, 2)), np.array([0, 1, 2]))


def test_logistic_regression_rejects_empty_dataset():
    model = LogisticRegressionGD()
    with pytest.raises(ValueError):
        model.fit(np.zeros((0, 2)), np.zeros(0))


def test_logistic_regression_predict_proba_before_fit_raises():
    model = LogisticRegressionGD()
    with pytest.raises(RuntimeError):
        model.predict_proba(np.zeros((2, 2)))


def test_logistic_regression_loss_decreases():
    rng = np.random.default_rng(2)
    n = 300
    X = rng.normal(0, 1, size=(n, 3))
    true_w = np.array([1.5, -1.0, 0.5])
    y = (X @ true_w + rng.normal(0, 0.5, n) > 0).astype(float)
    model = LogisticRegressionGD(learning_rate=0.3, n_iters=500).fit(X, y)
    # loss should generally trend down: first quarter avg > last quarter avg
    q = len(model.loss_history_) // 4
    first_quarter_avg = np.mean(model.loss_history_[:q])
    last_quarter_avg = np.mean(model.loss_history_[-q:])
    assert last_quarter_avg < first_quarter_avg


def test_logistic_regression_diverges_raises_floatingpointerror():
    # Logistic regression's gradient is bounded (the sigmoid saturates the
    # error term to [-1, 1]), so unlike plain linear regression it can't
    # blow up exponentially from a merely-too-large learning rate -- it
    # takes a genuinely extreme combination of feature scale and learning
    # rate to overflow float64 in a handful of iterations. That's still a
    # real (if unusual) input the guard clause needs to catch cleanly
    # rather than silently returning nan/inf weights.
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1e200, size=(50, 4))
    y = rng.integers(0, 2, size=50).astype(float)
    model = LogisticRegressionGD(learning_rate=1e200, n_iters=5, l2=0.0)
    with pytest.raises(FloatingPointError):
        model.fit(X, y)
