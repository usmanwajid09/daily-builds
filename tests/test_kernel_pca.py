"""Tests for kernel_pca.model.KernelPCA."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from kernel_pca.model import KernelPCA


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


def test_rings_separated_by_first_component():
    x, y = _rings()
    z = KernelPCA(n_components=2, gamma=0.5).fit_transform(x)
    assert z.shape == (len(y), 2)
    threshold = (z[y == 0, 0].mean() + z[y == 1, 0].mean()) / 2
    pred = (z[:, 0] > threshold).astype(int)
    if np.mean(pred == y) < 0.5:
        pred = 1 - pred
    assert float(np.mean(pred == y)) >= 0.95


def test_linear_kernel_matches_pca_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    kpca = KernelPCA(n_components=2, kernel="linear").fit(x)
    # Linear kernel PCA eigenvalues equal the eigenvalues of X^T X (centred).
    xc = x - x.mean(axis=0)
    expected = np.linalg.eigvalsh(xc.T @ xc)[::-1][:2]
    assert kpca.lambdas_ == pytest.approx(expected, rel=1e-6)


def test_transform_new_points_shape():
    x, y = _rings()
    kpca = KernelPCA(n_components=2, gamma=0.5).fit(x)
    rng = np.random.default_rng(1)
    x_new = rng.normal(size=(10, 2))
    z_new = kpca.transform(x_new)
    assert z_new.shape == (10, 2)
    assert np.all(np.isfinite(z_new))


def test_fit_transform_matches_transform():
    x, _ = _rings()
    kpca = KernelPCA(n_components=2, gamma=0.5)
    z1 = kpca.fit_transform(x)
    z2 = kpca.transform(x)
    assert np.allclose(z1, z2)


def test_explained_variance_ratio_sums_below_one():
    x, _ = _rings()
    kpca = KernelPCA(n_components=2, gamma=0.5).fit(x)
    ratio = kpca.explained_variance_ratio_
    assert len(ratio) == 2
    assert ratio.sum() <= 1.0 + 1e-12
    assert np.all(ratio > 0)
    assert np.all(np.diff(ratio) <= 1e-12)  # non-increasing


def test_lambdas_non_increasing():
    x, _ = _rings()
    kpca = KernelPCA(n_components=3, gamma=0.5).fit(x)
    assert np.all(np.diff(kpca.lambdas_) <= 1e-12)
    assert np.all(kpca.lambdas_ > 0)


def test_transform_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        KernelPCA().transform(np.zeros((5, 2)))


def test_1d_x_raises():
    with pytest.raises(ValueError, match="2-dimensional"):
        KernelPCA().fit(np.zeros(10))


def test_empty_x_raises():
    with pytest.raises(ValueError, match="empty"):
        KernelPCA().fit(np.zeros((0, 2)))


def test_too_many_components_raises():
    x, _ = _rings(n=20)
    with pytest.raises(ValueError, match="n_components"):
        KernelPCA(n_components=25).fit(x)


def test_bad_hyperparameters_raise():
    with pytest.raises(ValueError, match="n_components"):
        KernelPCA(n_components=0)
    with pytest.raises(ValueError, match="kernel"):
        KernelPCA(kernel="poly")
    with pytest.raises(ValueError, match="gamma"):
        KernelPCA(gamma=-1.0)


def test_wrong_feature_count_on_transform_raises():
    x, _ = _rings()
    kpca = KernelPCA(n_components=2, gamma=0.5).fit(x)
    with pytest.raises(ValueError, match="features"):
        kpca.transform(np.zeros((5, 3)))


def test_deterministic():
    x, _ = _rings()
    a = KernelPCA(n_components=2, gamma=0.5).fit_transform(x)
    b = KernelPCA(n_components=2, gamma=0.5).fit_transform(x)
    assert np.allclose(np.abs(a), np.abs(b))
