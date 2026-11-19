"""Demo: spectral clustering on two concentric rings.

Plain k-means on the raw coordinates cannot split concentric rings (both
rings share the same centre), but the spectral embedding makes them
linearly separable. Prints the cluster sizes and the accuracy against
the true ring labels (up to a label swap).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from spectral_clustering.model import SpectralClustering


def make_rings(seed: int = 42, n: int = 200):
    """Two concentric rings, inner radius ~1, outer radius ~3."""
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


def main():
    x, y = make_rings()
    model = SpectralClustering(n_clusters=2, gamma=2.0, random_state=0)
    labels = model.fit_predict(x)

    print(f"Samples: {len(y)}, clusters: 2, gamma: 2.0")
    sizes = np.bincount(labels).tolist()
    print(f"Cluster sizes: {sizes}")
    print("Smallest Laplacian eigenvalues:",
          np.round(model.eigenvalues_, 3).tolist())
    acc = float(np.mean(labels == y))
    acc = max(acc, 1 - acc)  # cluster labels are arbitrary up to a swap
    print(f"Ring recovery accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
