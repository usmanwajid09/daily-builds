"""Demo: impute randomly masked entries of a correlated 60x4 dataset with kNN."""
import os
import sys

# Running as `python knn_imputer/demo.py` puts the script's own folder,
# not the repo root, on sys.path - add the repo root so the package import
# below resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from knn_imputer.model import KNNImputer


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=60)
    # Four correlated features built from a shared latent factor.
    X = np.column_stack(
        [
            base,
            2.0 * base + rng.normal(scale=0.3, size=60),
            -1.0 * base + rng.normal(scale=0.3, size=60),
            0.5 * base + rng.normal(scale=0.5, size=60),
        ]
    )
    return X


def main():
    X_true = make_data()
    rng = np.random.default_rng(42)
    mask = rng.random(X_true.shape) < 0.20
    X_masked = X_true.copy()
    X_masked[mask] = np.nan
    n_imputed = int(mask.sum())

    for weights in ("uniform", "distance"):
        imputer = KNNImputer(n_neighbors=5, weights=weights)
        X_filled = imputer.fit_transform(X_masked)
        rmse = float(np.sqrt(np.mean((X_filled[mask] - X_true[mask]) ** 2)))
        print(
            f"weights={weights}: imputed {n_imputed} entries, "
            f"RMSE on masked entries = {rmse:.4f}"
        )

    # Before/after look at the first rows that had missing values.
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_filled = imputer.fit_transform(X_masked)
    rows = np.flatnonzero(mask.any(axis=1))[:4]
    print("\nBefore / after (first 4 rows with missing values, NaN = masked):")
    for i in rows:
        before = " ".join(f"{v:7.3f}" if not np.isnan(v) else "    nan" for v in X_masked[i])
        after = " ".join(f"{v:7.3f}" for v in X_filled[i])
        print(f"row {i:2d} masked: [{before}]")
        print(f"row {i:2d} filled: [{after}]")
    print(f"\nNaNs remaining after imputation: {int(np.isnan(X_filled).sum())}")


if __name__ == "__main__":
    main()
