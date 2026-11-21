"""Tests for pls_regression.PLSRegression."""

import numpy as np
import pytest

from pls_regression import PLSRegression


def make_latent(n=100, seed=0, n_latent=2, n_features=5, noise=0.0, x_noise=0.0):
    """X built from n_latent factors; y a linear function of them."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, n_latent))
    loadings = rng.normal(size=(n_latent, n_features))
    X = latent @ loadings + rng.normal(scale=x_noise, size=(n, n_features))
    beta = rng.normal(size=n_latent)
    y = latent @ beta + rng.normal(scale=noise, size=n)
    return X, y


def test_perfect_linear_relationship_recovered():
    """Noiseless low-rank data: PLS recovers y almost exactly."""
    X, y = make_latent()
    model = PLSRegression(n_components=2).fit(X, y)
    assert model.score(X, y) == pytest.approx(1.0, abs=1e-6)
    assert model.predict(X) == pytest.approx(y, abs=1e-4)


def test_single_component_on_rank1_data_is_near_exact():
    """With a single latent factor, one component explains everything."""
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(80, 1))
    X = latent @ rng.normal(size=(1, 6))
    y = 3.0 * latent[:, 0]
    model = PLSRegression(n_components=1).fit(X, y)
    assert model.score(X, y) == pytest.approx(1.0, abs=1e-9)


def test_coef_shape_1d_y():
    X, y = make_latent()
    model = PLSRegression(n_components=2).fit(X, y)
    assert model.coef_.shape == (5, 1)
    assert model.intercept_.shape == (1,)


def test_coef_shape_2d_y():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    Y = np.column_stack([X[:, 0] + 2 * X[:, 1], X[:, 2] - X[:, 3]])
    model = PLSRegression(n_components=2).fit(X, Y)
    assert model.coef_.shape == (4, 2)
    assert model.intercept_.shape == (2,)
    assert model.predict(X).shape == (60, 2)


def test_intercept_matches_data_mean():
    """Predicting at the training mean of X gives ~ the mean of y."""
    X, y = make_latent()
    model = PLSRegression(n_components=2).fit(X, y)
    pred_at_mean = model.predict(X.mean(axis=0, keepdims=True))
    assert pred_at_mean == pytest.approx(y.mean(), abs=1e-9)


def test_transform_returns_scores():
    X, y = make_latent()
    model = PLSRegression(n_components=2).fit(X, y)
    scores = model.transform(X)
    assert scores.shape == (100, 2)
    # Scores on the training data match the stored X-scores.
    assert scores == pytest.approx(model.x_scores_, abs=1e-9)


def test_score_is_one_for_noiseless_data():
    X, y = make_latent()
    model = PLSRegression(n_components=2).fit(X, y)
    assert model.score(X, y) == pytest.approx(1.0, abs=1e-6)


def test_more_components_never_hurt_training_r2():
    """Training R^2 is (essentially) non-decreasing in n_components."""
    X, y = make_latent(noise=0.5)
    r2s = [
        PLSRegression(n_components=k).fit(X, y).score(X, y) for k in (1, 2, 3, 4)
    ]
    for prev, nxt in zip(r2s, r2s[1:]):
        assert nxt >= prev - 1e-8


def test_predict_before_fit_raises():
    X, _ = make_latent()
    with pytest.raises(ValueError, match="not fitted"):
        PLSRegression().predict(X)
    with pytest.raises(ValueError, match="not fitted"):
        PLSRegression().transform(X)
    with pytest.raises(ValueError, match="not fitted"):
        PLSRegression().score(X, np.zeros(100))


def test_n_components_zero_raises():
    with pytest.raises(ValueError, match="n_components"):
        PLSRegression(n_components=0)


def test_n_components_too_large_raises():
    X, y = make_latent(n=20)
    with pytest.raises(ValueError, match="too large"):
        PLSRegression(n_components=10).fit(X, y)


def test_empty_X_raises():
    with pytest.raises(ValueError, match="non-empty"):
        PLSRegression().fit(np.empty((0, 3)), np.empty(0))


def test_mismatched_X_y_lengths_raises():
    X, y = make_latent()
    with pytest.raises(ValueError, match="mismatched"):
        PLSRegression().fit(X, y[:-1])


def test_nan_in_X_raises():
    X, y = make_latent()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        PLSRegression().fit(X, y)


def test_nan_in_y_raises():
    X, y = make_latent()
    y = y.copy()
    y[3] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        PLSRegression().fit(X, y)


def test_1d_X_raises():
    X, y = make_latent()
    with pytest.raises(ValueError, match="2D"):
        PLSRegression().fit(X[:, 0], y)


def test_constant_column_with_scale_does_not_crash():
    """A zero-variance column is left unscaled (scale factor 1)."""
    X, y = make_latent()
    X = np.column_stack([X, np.full(len(y), 7.0)])
    model = PLSRegression(n_components=2, scale=True).fit(X, y)
    assert model.score(X, y) == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(model.coef_))


def test_multi_output_noiseless():
    """Two targets, both recovered near-exactly."""
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(80, 2))
    X = latent @ rng.normal(size=(2, 5))
    Y = np.column_stack([2 * latent[:, 0] - latent[:, 1],
                         latent[:, 0] + 3 * latent[:, 1]])
    model = PLSRegression(n_components=2).fit(X, Y)
    assert model.score(X, Y) == pytest.approx(1.0, abs=1e-6)
    assert model.predict(X) == pytest.approx(Y, abs=1e-4)


def test_reproducibility():
    """Fitting twice on the same data gives identical coefficients."""
    X, y = make_latent()
    c1 = PLSRegression(n_components=2).fit(X, y).coef_
    c2 = PLSRegression(n_components=2).fit(X, y).coef_
    assert c1 == pytest.approx(c2, abs=0.0)


def test_scale_false_still_fits():
    X, y = make_latent(noise=0.3)
    model = PLSRegression(n_components=2, scale=False).fit(X, y)
    # Without scaling, high-variance features dominate the weights, so
    # the fit is weaker than the scaled default — it should still
    # capture most of the signal.
    assert model.score(X, y) > 0.7


def test_bad_max_iter_and_tol_raise():
    with pytest.raises(ValueError, match="max_iter"):
        PLSRegression(max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        PLSRegression(tol=0.0)
