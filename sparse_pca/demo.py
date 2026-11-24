"""Demo: Sparse PCA on synthetic data with a known sparse signal.

Only the first 3 of 10 features carry the signal: they share one strong
latent factor, while the other 7 are pure noise. A good sparse PCA should
recover a leading component concentrated on features 0-2, with the noise
features thresholded to exactly zero, and still reconstruct the data well.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sparse_pca.model import SparsePCA


def make_data(seed: int = 0, n: int = 400):
    """10 features; only the first 3 share a strong latent factor."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=n)  # shared latent factor
    X = rng.normal(scale=1.0, size=(n, 10))
    X[:, :3] += z[:, None]
    return X


def main():
    X = make_data()
    print(f"Data: {X.shape[0]} samples x {X.shape[1]} features "
          f"(signal on features 0-2 only)")

    pca = SparsePCA(n_components=2, alpha=0.5, random_state=0)
    pca.fit(X)

    loadings = pca.components_[0]
    print("\nLeading component loadings:")
    print("  " + " ".join(f"{v:+.3f}" for v in loadings))
    print(f"Fraction of zero loadings (all components): {pca.sparsity_:.2f}")
    print(f"Explained variance: {pca.explained_variance_}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    X_rec = pca.inverse_transform(pca.transform(X))
    mse = float(np.mean((X - X_rec) ** 2))
    print(f"Reconstruction MSE (2 components): {mse:.4f}")

    # -- checks that must hold -------------------------------------------
    top3 = set(np.argsort(-np.abs(loadings))[:3].tolist())
    assert top3 == {0, 1, 2}, f"top loadings not on signal features: {top3}"
    assert pca.sparsity_ > 0, "expected some loadings thresholded to zero"
    assert np.allclose(np.linalg.norm(pca.components_, axis=1), 1.0), \
        "components must be unit-norm"
    assert pca.explained_variance_ratio_[0] > 0.5, \
        "leading component should explain > 50% of the variance"
    assert mse < 1.5, f"reconstruction MSE too high: {mse:.4f}"
    print("\nAll demo checks passed.")


if __name__ == "__main__":
    main()
