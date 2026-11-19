"""Demo: classical MDS recovers a known 2D layout.

Samples 30 points from three well-separated 2D clusters, forgets the
coordinates, keeps only the pairwise distances, and asks MDS to
recover the layout. The recovered embedding is compared to the true
coordinates with a Procrustes-style check (correlation after optimal
rotation), and the Kruskal stress is reported.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mds_embedding.model import ClassicalMDS, pairwise_distances


def make_clusters(seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    points = np.vstack(
        [c + rng.normal(0, 0.6, size=(10, 2)) for c in centres]
    )
    return points


def procrustes_correlation(true, recovered):
    """Correlation between true and recovered coords after optimal rotation."""
    a = true - true.mean(axis=0)
    b = recovered - recovered.mean(axis=0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    aligned = b @ r
    return float(np.corrcoef(a.ravel(), aligned.ravel())[0, 1])


def main():
    points = make_clusters()
    d = pairwise_distances(points)
    model = ClassicalMDS(n_components=2)
    embedding = model.fit_transform(d)
    print(f"Points: {points.shape[0]}, embedding shape: {embedding.shape}")
    print(f"Eigenvalues: {np.round(model.eigenvalues_, 3).tolist()}")
    print(f"Kruskal stress-1: {model.stress_:.4f}")
    print(f"Procrustes correlation with true layout: "
          f"{procrustes_correlation(points, embedding):.4f}")
    print("First 5 recovered points:")
    for row in np.round(embedding[:5], 3):
        print(f"  {row.tolist()}")


if __name__ == "__main__":
    main()
