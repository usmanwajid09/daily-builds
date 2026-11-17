"""Demo: factorize a synthetic non-negative matrix with NMF.

Builds V = W_true @ H_true (seed 42, shape 20x12, rank 3, uniform
entries), factorizes it with n_components=3, and prints the original
shape, the relative reconstruction error, and the iteration count.
Prints to stdout only; writes no files.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nmf.model import NMF
import numpy as np


def main():
    rng = np.random.default_rng(42)
    W_true = rng.uniform(0, 1, size=(20, 3))
    H_true = rng.uniform(0, 1, size=(3, 12))
    V = W_true @ H_true

    model = NMF(n_components=3, max_iter=500, tol=1e-4, random_state=42)
    W = model.fit_transform(V)
    recon = W @ model.components_
    rel_err = np.linalg.norm(V - recon) / np.linalg.norm(V)

    print(f"Original shape: {V.shape}")
    print(f"Relative reconstruction error: {rel_err:.6f}")
    print(f"Iterations: {model.n_iter_}")
    assert rel_err < 0.05, f"relative error {rel_err:.6f} exceeds 0.05"


if __name__ == "__main__":
    main()
