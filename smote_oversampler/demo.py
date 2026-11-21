"""Imbalanced two-blob demo: 100 majority vs 10 minority samples in 2D.

Runs SMOTE (seed 42), prints the class counts before and after, how
many synthetic points were created, and checks that the synthetic
points sit close to real minority points (they lie on segments between
real minority samples, so they never stray far from the minority blob).
"""
import os
import sys

# Running as `python smote_oversampler/demo.py` puts the package directory,
# not the repo root, on sys.path; add the repo root so the package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from smote_oversampler.model import SMOTE


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    majority = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(100, 2))
    minority = rng.normal(loc=[4.0, 4.0], scale=1.0, size=(10, 2))
    X = np.vstack([majority, minority])
    y = np.array([0] * 100 + [1] * 10)
    return X, y


def main():
    X, y = make_data()
    _, before = np.unique(y, return_counts=True)
    print(f"Before SMOTE: class 0 = {before[0]}, class 1 = {before[1]} "
          f"(total {len(y)})")

    smote = SMOTE(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    _, after = np.unique(y_res, return_counts=True)
    print(f"After SMOTE:  class 0 = {after[0]}, class 1 = {after[1]} "
          f"(total {len(y_res)})")
    print(f"Synthetic samples created: {smote.n_synthetic_}")

    # Synthetic points are the rows appended after the original 110.
    synthetic = X_res[len(X):]
    real_minority = X[y == 1]
    dists = np.sqrt(((synthetic[:, None, :] - real_minority[None, :, :]) ** 2).sum(axis=2))
    nearest = dists.min(axis=1)
    print(f"Distance from each synthetic point to its nearest real "
          f"minority sample: min {nearest.min():.3f}, max {nearest.max():.3f}")
    print("First 3 synthetic points:")
    for point in synthetic[:3]:
        print(f"  [{point[0]:.3f}, {point[1]:.3f}]")


if __name__ == "__main__":
    main()
