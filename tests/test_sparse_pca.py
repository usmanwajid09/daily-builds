"""Tests for sparse_pca.model.SparsePCA."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from sparse_pca.model import SparsePCA


def _signal_data(seed=0, n=400):
    """10 features; only the first 3 share a strong latent factor."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=n)
    X = rng.normal(scale=1.0, size=(n, 10))
    X[:, :3] += z[:, None]
    return X


def _plain_pca_top_direction(X):
    """Top eigenvector of the sample covariance (plain PCA direction)."""
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (len(X) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, np.argmax(eigvals)]


def _zero_fraction(pca):
    return float(np.mean(pca.components_ == 0.0))


# -- core behaviour -------------------------------------------------------

def test_alpha_zero_approximates_plain_pca():
    X = _signal_data()
    pca = SparsePCA(n_components=1, alpha=0.0, random_state=0).fit(X)
    top = _plain_pca_top_direction(X)
    cosine = abs(float(pca.components_[0] @ top))
    assert cosine > 0.99, f"|cosine| with plain PCA direction = {cosine:.4f}"


def test_sparsity_increases_with_alpha():
    X = _signal_data()
    low = SparsePCA(n_components=2, alpha=0.0, random_state=0).fit(X)
    high = SparsePCA(n_components=2, alpha=1.0, random_state=0).fit(X)
    assert _zero_fraction(high) > _zero_fraction(low)
    assert _zero_fraction(high) > 0


def test_components_unit_norm():
    X = _signal_data()
    pca = SparsePCA(n_components=3, alpha=0.5, random_state=1).fit(X)
    norms = np.linalg.norm(pca.components_, axis=1)
    assert pca.components_.shape == (3, 10)
    assert np.allclose(norms, 1.0)


def test_transform_shape():
    X = _signal_data()
    pca = SparsePCA(n_components=2, alpha=0.5, random_state=0).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (len(X), 2)
    assert np.isfinite(Z).all()


def test_reconstruction_error_decreases_with_more_components():
    X = _signal_data()
    errors = []
    for k in (1, 2, 4):
        pca = SparsePCA(n_components=k, alpha=0.5, random_state=0).fit(X)
        rec = pca.inverse_transform(pca.transform(X))
        errors.append(float(np.mean((X - rec) ** 2)))
    assert errors[0] > errors[1] > errors[2], f"MSEs: {errors}"


def test_explained_variance_positive_and_nonincreasing():
    X = _signal_data()
    pca = SparsePCA(n_components=3, alpha=0.5, random_state=0).fit(X)
    ev = pca.explained_variance_
    assert (ev > 0).all()
    assert np.all(np.diff(ev) <= 1e-9), f"explained variance: {ev}"
    assert np.allclose(pca.explained_variance_ratio_.sum(),
                       ev.sum() / np.var(X, axis=0, ddof=1).sum())


def test_sparse_leading_component_finds_signal_features():
    X = _signal_data()
    pca = SparsePCA(n_components=1, alpha=0.5, random_state=0).fit(X)
    top3 = set(np.argsort(-np.abs(pca.components_[0]))[:3].tolist())
    assert top3 == {0, 1, 2}
    assert _zero_fraction(pca) > 0


def test_determinism_with_same_random_state():
    X = _signal_data()
    a = SparsePCA(n_components=2, alpha=0.5, random_state=7).fit(X)
    b = SparsePCA(n_components=2, alpha=0.5, random_state=7).fit(X)
    assert np.allclose(a.components_, b.components_)
    assert np.allclose(a.explained_variance_, b.explained_variance_)


def test_fit_returns_self_and_sets_attributes():
    X = _signal_data()
    pca = SparsePCA(n_components=2, alpha=0.5, random_state=0)
    assert pca.fit(X) is pca
    assert pca.mean_.shape == (10,)
    assert pca.sparsity_ == pytest.approx(_zero_fraction(pca))


# -- validation errors ----------------------------------------------------

def test_value_error_1d_input():
    with pytest.raises(ValueError):
        SparsePCA().fit(np.zeros(10))


def test_value_error_empty_input():
    with pytest.raises(ValueError):
        SparsePCA().fit(np.empty((0, 5)))


def test_value_error_nan_input():
    X = _signal_data()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        SparsePCA().fit(X)


def test_value_error_n_components_less_than_one():
    with pytest.raises(ValueError):
        SparsePCA(n_components=0)


def test_value_error_n_components_too_large():
    X = _signal_data()
    with pytest.raises(ValueError):
        SparsePCA(n_components=11).fit(X)


def test_value_error_negative_alpha():
    with pytest.raises(ValueError):
        SparsePCA(alpha=-0.1)


def test_value_error_transform_before_fit():
    X = _signal_data()
    with pytest.raises(ValueError):
        SparsePCA().transform(X)
    with pytest.raises(ValueError):
        SparsePCA().inverse_transform(np.zeros((5, 2)))
