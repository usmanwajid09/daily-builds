"""Generates low-rank synthetic data, fits truncated SVD, reports variance explained and reconstruction error."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truncated_svd.model import TruncatedSVD


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n, d, rank = 200, 10, 3
    # Rank-3 signal plus small noise.
    u = rng.normal(size=(n, rank))
    v = rng.normal(size=(rank, d))
    x = u @ v + rng.normal(scale=0.1, size=(n, d))
    return x


def main():
    x = make_data()
    model = TruncatedSVD(n_components=3).fit(x)
    projected = model.transform(x)
    mse = model.reconstruction_error(x)
    # Compare with keeping only 1 component.
    mse1 = TruncatedSVD(n_components=1).fit(x).reconstruction_error(x)

    print(f"Samples: {x.shape[0]}, features: {x.shape[1]}, components kept: 3")
    print(f"Singular values: {np.round(model.singular_values_, 3).tolist()}")
    print(f"Explained variance ratios: {np.round(model.explained_variance_ratio_, 3).tolist()}")
    print(f"Total variance explained by 3 components: {float(model.explained_variance_ratio_.sum()):.3f}")
    print(f"Reconstruction MSE with 3 components: {mse:.4f}")
    print(f"Reconstruction MSE with 1 component:  {mse1:.4f}")
    print(f"Projected shape: {projected.shape}")


if __name__ == "__main__":
    main()
