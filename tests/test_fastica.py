"""Tests for fastica (FastICA, numpy only)."""

import numpy as np
import pytest

from fastica.model import FastICA


def make_mixture(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    s1 = rng.uniform(-1, 1, n)
    s2 = rng.laplace(0, 1, n)
    S = np.column_stack([s1, s2])
    A = np.array([[1.0, 1.0], [0.5, 2.0]])
    return S, S @ A.T, A


def best_match_corr(S, S_hat):
    corrs = np.corrcoef(np.column_stack([S, S_hat]), rowvar=False)
    n = S.shape[1]
    return float(np.abs(corrs[:n, n:]).max(axis=1).mean())


def test_recovers_sources():
    """Recovered components correlate > 0.9 with the true sources."""
    S, X, _ = make_mixture()
    S_hat = FastICA(n_components=2, random_state=42).fit_transform(X)
    assert best_match_corr(S, S_hat) > 0.9


def test_inverse_transform_round_trip():
    S, X, _ = make_mixture()
    ica = FastICA(n_components=2, random_state=42)
    S_hat = ica.fit_transform(X)
    X_rec = ica.inverse_transform(S_hat)
    rmse = float(np.sqrt(np.mean((X - X_rec) ** 2)))
    assert rmse < 1e-8


def test_transform_shape():
    S, X, _ = make_mixture()
    ica = FastICA(n_components=2, random_state=42).fit(X)
    out = ica.transform(X)
    assert out.shape == (X.shape[0], 2)


def test_fit_transform_consistency():
    S, X, _ = make_mixture()
    ica1 = FastICA(n_components=2, random_state=7)
    a = ica1.fit_transform(X)
    ica2 = FastICA(n_components=2, random_state=7)
    ica2.fit(X)
    b = ica2.transform(X)
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)


def test_n_components_one():
    S, X, _ = make_mixture()
    ica = FastICA(n_components=1, random_state=42).fit(X)
    assert ica.transform(X).shape == (X.shape[0], 1)
    assert ica.components_.shape == (1, 2)
    assert ica.mixing_.shape == (2, 1)


def test_determinism_same_seed():
    S, X, _ = make_mixture()
    a = FastICA(n_components=2, random_state=123).fit_transform(X)
    b = FastICA(n_components=2, random_state=123).fit_transform(X)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        FastICA().fit(np.empty((0, 2)))


def test_1d_input_raises():
    with pytest.raises(ValueError):
        FastICA().fit(np.zeros(10))


def test_wrong_n_components_raises():
    S, X, _ = make_mixture()
    with pytest.raises(ValueError):
        FastICA(n_components=3).fit(X)
    with pytest.raises(ValueError):
        FastICA(n_components=0)
    with pytest.raises(ValueError):
        FastICA(n_components=-1)


def test_nan_input_raises():
    X = np.zeros((10, 2))
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        FastICA().fit(X)
    X[0, 0] = np.inf
    with pytest.raises(ValueError):
        FastICA().fit(X)


def test_not_fitted_transform_raises():
    S, X, _ = make_mixture()
    with pytest.raises((RuntimeError, ValueError)):
        FastICA().transform(X)
    with pytest.raises((RuntimeError, ValueError)):
        FastICA().inverse_transform(np.zeros((5, 2)))


def test_non_numeric_input_raises():
    with pytest.raises((TypeError, ValueError)):
        FastICA().fit(np.array([["a", "b"], ["c", "d"]]))


def test_fit_returns_self_and_attributes():
    S, X, _ = make_mixture()
    ica = FastICA(n_components=2, random_state=42)
    assert ica.fit(X) is ica
    assert ica.components_.shape == (2, 2)
    assert ica.mixing_.shape == (2, 2)
    assert ica.mean_.shape == (2,)
    assert ica.n_iter_ >= 1
