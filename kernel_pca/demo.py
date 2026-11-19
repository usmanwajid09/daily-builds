"""Demo: kernel PCA on two concentric rings.

Plain PCA cannot separate concentric rings (the classes differ in radius,
not direction), but the RBF kernel maps them to a space where the first
kernel component separates them cleanly. Prints the explained variance
ratio and the separation of the two rings along the first component.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from kernel_pca.model import KernelPCA


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
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5)
    z = kpca.fit_transform(x)

    print(f"Samples: {len(y)}, components: 2, kernel: rbf (gamma=0.5)")
    print("Explained variance ratio:",
          np.round(kpca.explained_variance_ratio_, 3).tolist())
    print("First component range:",
          f"[{z[:, 0].min():.3f}, {z[:, 0].max():.3f}]")
    for label in (0, 1):
        mean = float(z[y == label, 0].mean())
        print(f"  ring {label}: mean first component = {mean:.3f}")
    # A threshold on the first component separates the rings.
    threshold = float((z[y == 0, 0].mean() + z[y == 1, 0].mean()) / 2)
    pred = (z[:, 0] > threshold).astype(int)
    if np.mean(pred == y) < 0.5:  # sign of the component is arbitrary
        pred = 1 - pred
    print(f"Ring separation accuracy on component 1: "
          f"{float(np.mean(pred == y)):.3f}")


if __name__ == "__main__":
    main()
