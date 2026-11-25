"""Demo: FastICA recovers two mixed non-Gaussian sources.

Two independent sources (uniform and Laplace) are mixed with a fixed 2x2
matrix. FastICA should recover them up to permutation and sign, so the
headline number is the mean best-match |correlation| between the true
sources and the recovered components (1.0 = perfect recovery), plus the
reconstruction error of the round trip through inverse_transform.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fastica.model import FastICA


def make_mixture(seed: int = 0, n: int = 2000):
    """Two independent non-Gaussian sources, mixed with a fixed matrix."""
    rng = np.random.default_rng(seed)
    s1 = rng.uniform(-1, 1, n)               # uniform (sub-Gaussian)
    s2 = rng.laplace(0, 1, n)                # Laplace (super-Gaussian)
    S = np.column_stack([s1, s2])
    A = np.array([[1.0, 1.0], [0.5, 2.0]])   # mixing matrix
    X = S @ A.T
    return S, X, A


def best_match_corr(S, S_hat):
    """Mean best-match |corr| between true and recovered sources.

    For each true source, take the largest |correlation| over the recovered
    components (handles permutation and sign flips).
    """
    corrs = np.corrcoef(np.column_stack([S, S_hat]), rowvar=False)
    n = S.shape[1]
    cross = np.abs(corrs[:n, n:])
    return float(cross.max(axis=1).mean())


def main():
    S, X, A = make_mixture()
    ica = FastICA(n_components=2, random_state=42)
    S_hat = ica.fit_transform(X)

    mean_corr = best_match_corr(S, S_hat)
    X_rec = ica.inverse_transform(S_hat)
    recon_rmse = float(np.sqrt(np.mean((X - X_rec) ** 2)))
    print(f"FastICA: mean best-match |corr| with true sources = {mean_corr:.4f}")
    print(f"Reconstruction RMSE (inverse_transform round trip) = {recon_rmse:.2e}")
    print(f"Converged in {ica.n_iter_} iterations "
          f"(max_iter={ica.max_iter})")
    print(f"Recovered mixing matrix:\n{np.round(ica.mixing_, 3)}")
    print(f"True mixing matrix:\n{A}")


if __name__ == "__main__":
    main()
