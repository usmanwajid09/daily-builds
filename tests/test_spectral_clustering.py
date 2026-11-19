"""Tests for spectral_clustering.model.SpectralClustering."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from spectral_clustering.model import SpectralClustering


def _rings(seed=0, n=120):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta = rng.uniform(0, 2 * np.pi, n)
    radii = np.concatenate([
        rng.normal(loc=1.0, scale=0.15, size=n_half),
        rng.normal(loc=3.0, scale=0.15, size=n_half),
    ])
    x = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    y = np.array([0] * n_half + [1] * n_half)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def _blobs(seed=0, n=90):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k, loc in enumerate([(0, 0), (5, 0), (2.5, 4)]):
        xs.append(rng.normal(loc=loc, scale=0.6, size=(n // 3, 2)))
        ys.extend([k] * (n // 3))
    x = np.vstack(xs)
    y = np.array(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def _recovery_accuracy(labels, y):
    """Best accuracy over all label permutations (small cluster counts)."""
    from itertools import permutations
    best = 0.0
    for perm in permutations(np.unique(labels)):
        mapping = {old: new for old, new in zip(np.unique(labels), perm)}
        mapped = np.array([mapping[v] for v in labels])
        best = max(best, float(np.mean(mapped == y)))
    return best


def test_rings_recovered():
    x, y = _rings()
    labels = SpectralClustering(n_clusters=2, gamma=2.0,
                                random_state=0).fit_predict(x)
    assert _recovery_accuracy(labels, y) >= 0.95


def test_blobs_recovered():
    x, y = _blobs()
    labels = SpectralClustering(n_clusters=3, gamma=2.0,
                                random_state=0).fit_predict(x)
    assert _recovery_accuracy(labels, y) >= 0.95


def test_labels_shape_and_values():
    x, _ = _rings()
    model = SpectralClustering(n_clusters=2, gamma=2.0, random_state=0)
    labels = model.fit_predict(x)
    assert labels.shape == (len(x),)
    assert set(np.unique(labels)) == {0, 1}
    assert np.array_equal(model.labels_, labels)


def test_embedding_shape():
    x, _ = _rings()
    model = SpectralClustering(n_clusters=2, gamma=2.0,
                               random_state=0).fit(x)
    assert model.embedding_.shape == (len(x), 2)
    assert np.all(np.isfinite(model.embedding_))
    # Rows are normalised to unit length.
    norms = np.linalg.norm(model.embedding_, axis=1)
    assert np.allclose(norms, 1.0)


def test_eigenvalues_sorted():
    x, _ = _rings()
    model = SpectralClustering(n_clusters=2, gamma=2.0,
                               random_state=0).fit(x)
    assert len(model.eigenvalues_) == 2
    assert model.eigenvalues_[0] <= model.eigenvalues_[1] + 1e-12
    assert model.eigenvalues_[0] >= -1e-9


def test_reproducible_with_random_state():
    x, _ = _rings()
    a = SpectralClustering(n_clusters=2, gamma=2.0,
                           random_state=7).fit_predict(x)
    b = SpectralClustering(n_clusters=2, gamma=2.0,
                           random_state=7).fit_predict(x)
    assert np.array_equal(a, b)


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2-dimensional"):
        SpectralClustering().fit(np.zeros(10))


def test_empty_x_raises():
    with pytest.raises(ValueError, match="empty"):
        SpectralClustering().fit(np.zeros((0, 2)))


def test_too_few_samples_raises():
    x = np.zeros((2, 2))
    with pytest.raises(ValueError, match="more samples than clusters"):
        SpectralClustering(n_clusters=2).fit(x)


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_clusters"):
        SpectralClustering(n_clusters=1)
    with pytest.raises(ValueError, match="gamma"):
        SpectralClustering(gamma=-1.0)
    with pytest.raises(ValueError, match="assign_labels"):
        SpectralClustering(assign_labels="discretize")


def test_inertia_is_positive():
    x, _ = _rings()
    model = SpectralClustering(n_clusters=2, gamma=2.0,
                               random_state=0).fit(x)
    assert model.inertia_ >= 0
    assert np.isfinite(model.inertia_)
