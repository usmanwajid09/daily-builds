"""Tests for VotingRegressor and its numpy-only base regressors."""

import numpy as np
import pytest

from voting_regressor.model import (
    DecisionTreeRegressor,
    KNNRegressor,
    RidgeRegressor,
    VotingRegressor,
)


def make_data(seed=42, n=120, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, noise, size=n)
    return X, y


def make_linear_data(seed=7, n=100):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.05, size=n)
    return X, y


def base_predictions(model, X):
    return np.column_stack([est.predict(X) for est in model.estimators_])


def test_predictions_shape_and_finite():
    X, y = make_data()
    model = VotingRegressor(voting="mean").fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.all(np.isfinite(preds))


def test_mean_voting_equals_average_of_base_predictions():
    X, y = make_data()
    model = VotingRegressor(voting="mean").fit(X, y)
    expected = base_predictions(model, X).mean(axis=1)
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-12)


def test_median_voting_equals_per_sample_median():
    X, y = make_data()
    model = VotingRegressor(voting="median").fit(X, y)
    expected = np.median(base_predictions(model, X), axis=1)
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-12)


def test_weighted_voting_equals_weighted_average():
    X, y = make_data()
    weights = [0.5, 0.3, 0.2]
    model = VotingRegressor(voting="weighted", weights=weights).fit(X, y)
    expected = base_predictions(model, X) @ np.array(weights)
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-12)


def test_weights_normalised_to_sum_one():
    X, y = make_data()
    model = VotingRegressor(voting="weighted", weights=[2, 2, 4]).fit(X, y)
    np.testing.assert_allclose(model.weights_, [0.25, 0.25, 0.5])
    assert abs(model.weights_.sum() - 1.0) < 1e-12
    # None means equal weights.
    equal = VotingRegressor(voting="weighted").fit(X, y)
    np.testing.assert_allclose(equal.weights_, [1 / 3, 1 / 3, 1 / 3])


def test_weights_wrong_length_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="length 3"):
        VotingRegressor(voting="weighted", weights=[0.5, 0.5]).fit(X, y)


def test_weights_negative_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="non-negative"):
        VotingRegressor(voting="weighted", weights=[0.5, -0.1, 0.6]).fit(X, y)


def test_weights_all_zero_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="all zero"):
        VotingRegressor(voting="weighted", weights=[0, 0, 0]).fit(X, y)


def test_unknown_voting_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="voting must be one of"):
        VotingRegressor(voting="majority").fit(X, y)


def test_predict_before_fit_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="not fitted"):
        VotingRegressor().predict(X)
    with pytest.raises(ValueError, match="not fitted"):
        RidgeRegressor().predict(X)
    with pytest.raises(ValueError, match="not fitted"):
        KNNRegressor().predict(X)
    with pytest.raises(ValueError, match="not fitted"):
        DecisionTreeRegressor().predict(X)


def test_x_must_be_2d():
    X, y = make_data()
    with pytest.raises(ValueError, match="2D"):
        VotingRegressor().fit(X[:, 0], y)


def test_xy_length_mismatch_raises():
    X, y = make_data()
    with pytest.raises(ValueError, match="inconsistent lengths"):
        VotingRegressor().fit(X, y[:-5])


def test_fit_returns_self():
    X, y = make_data()
    model = VotingRegressor()
    assert model.fit(X, y) is model
    assert len(model.estimators_) == 3
    assert model.n_features_in_ == 2
    assert model.weights_.shape == (3,)


def test_fits_linear_target_with_low_rmse():
    X, y = make_linear_data()
    split = int(0.8 * len(y))
    model = VotingRegressor(voting="mean").fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    rmse = float(np.sqrt(np.mean((y[split:] - preds) ** 2)))
    assert rmse < 0.5


def test_median_robust_to_one_bad_base_learner():
    """With one base learner deliberately corrupted, median voting stays
    closer to the truth than mean voting (a simple behaviour check)."""
    X, y = make_data()
    split = int(0.8 * len(y))
    X_test, y_test = X[split:], y[split:]

    class BadRegressor:
        """A stub base learner that always predicts wildly wrong values."""

        def predict(self, X):
            return np.full(len(X), 1_000.0)

    mean_model = VotingRegressor(voting="mean").fit(X[:split], y[:split])
    median_model = VotingRegressor(voting="median").fit(X[:split], y[:split])
    # Corrupt the tree (third learner) in both ensembles the same way.
    mean_model.estimators_[2] = BadRegressor()
    median_model.estimators_[2] = BadRegressor()

    mean_err = np.mean(np.abs(mean_model.predict(X_test) - y_test))
    median_err = np.mean(np.abs(median_model.predict(X_test) - y_test))
    assert median_err < mean_err
    # Median should be close to the uncorrupted ensemble's error.
    clean = VotingRegressor(voting="median").fit(X[:split], y[:split])
    clean_err = np.mean(np.abs(clean.predict(X_test) - y_test))
    assert median_err < 5 * clean_err
