"""Tests for mds_embedding."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mds_embedding import (
    ClassicalMDS,
    double_centre,
    kruskal_stress,
    pairwise_distances,
)
from mds_embedding.demo import make_clusters, procrustes_correlation


def test_pairwise_distances_known():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = pairwise_distances(x)
    assert d.shape == (2, 2)
    assert d[0, 1] == pytest.approx(5.0)
    assert d[0, 0] == 0.0
    assert np.allclose(d, d.T)


def test_pairwise_distances_validation():
    with pytest.raises(ValueError, match="2D"):
        pairwise_distances(np.zeros(4))
    with pytest.raises(ValueError, match="empty"):
        pairwise_distances(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="NaN"):
        pairwise_distances(np.array([[0.0, np.nan]]))


def test_double_centre_known():
    # Two points 2 apart: B should be [[1, -1], [-1, 1]].
    d = np.array([[0.0, 2.0], [2.0, 0.0]])
    b = double_centre(d)
    assert np.allclose(b, [[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(b.sum(axis=0), 0.0, atol=1e-12)  # rows sum to zero


def test_double_centre_validation():
    with pytest.raises(ValueError, match="square"):
        double_centre(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="negative"):
        double_centre(np.array([[0.0, -1.0], [-1.0, 0.0]]))


def test_mds_recovers_2d_layout():
    points = make_clusters()
    model = ClassicalMDS(n_components=2)
    emb = model.fit_transform(pairwise_distances(points))
    assert emb.shape == (30, 2)
    assert model.stress_ < 0.05
    assert procrustes_correlation(points, emb) > 0.99


def test_mds_recovers_1d_line():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=(15, 1))
    model = ClassicalMDS(n_components=1).fit(pairwise_distances(x))
    assert model.stress_ < 1e-8
    assert np.corrcoef(x.ravel(), model.embedding_.ravel())[0, 1] ** 2 > 0.999


def test_mds_eigenvalues_descending_and_positive():
    points = make_clusters()
    model = ClassicalMDS(n_components=2).fit(pairwise_distances(points))
    assert model.eigenvalues_.shape == (2,)
    assert model.eigenvalues_[0] >= model.eigenvalues_[1] > 0


def test_mds_from_points_classmethod():
    points = make_clusters()
    model = ClassicalMDS.from_points(points, n_components=2)
    assert model.embedding_.shape == (30, 2)
    assert model.stress_ < 0.05


def test_kruskal_stress_zero_for_exact_embedding():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    d = pairwise_distances(points)
    assert kruskal_stress(d, points) == pytest.approx(0.0)


def test_kruskal_stress_validation():
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="square"):
        kruskal_stress(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="one row per point"):
        kruskal_stress(d, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="all input distances are zero"):
        kruskal_stress(np.zeros((2, 2)), np.zeros((2, 2)))


def test_fit_validation():
    with pytest.raises(ValueError, match="square"):
        ClassicalMDS().fit(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="at least 2 points"):
        ClassicalMDS().fit(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="n_components"):
        ClassicalMDS(n_components=5).fit(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        ClassicalMDS(n_components=1).fit(np.array([[0.0, 1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="diagonal"):
        ClassicalMDS(n_components=1).fit(np.array([[1.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="n_components"):
        ClassicalMDS(n_components=0)


def test_non_euclidean_distances_raise():
    # Distances violating the triangle inequality are not Euclidean.
    d = np.array([[0.0, 1.0, 10.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="not Euclidean"):
        ClassicalMDS(n_components=2).fit(d)
