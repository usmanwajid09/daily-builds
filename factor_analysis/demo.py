"""Demo: factor analysis (EM) on synthetic data with a known 2-factor structure.

Generates 300 samples of 6 features from two latent factors with known
loadings, fits FactorAnalysis, and prints the recovered loadings next to
the true ones (up to rotation/sign), the reconstruction MSE and the
final log-likelihood.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from factor_analysis.model import FactorAnalysis

SEED = 42
N_SAMPLES = 300

# True loadings (6 features x 2 factors).
TRUE_W = np.array([
    [0.9, 0.1],
    [0.8, 0.2],
    [0.7, -0.1],
    [0.1, 0.9],
    [0.2, 0.8],
    [-0.1, 0.7],
])
TRUE_PSI = np.array([0.2, 0.3, 0.25, 0.2, 0.3, 0.25])
TRUE_MU = np.array([1.0, -0.5, 0.0, 0.5, -1.0, 0.25])


def make_data(seed: int = SEED):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(N_SAMPLES, 2))
    eps = rng.normal(size=(N_SAMPLES, 6)) * np.sqrt(TRUE_PSI)
    return z @ TRUE_W.T + TRUE_MU + eps


def main():
    x = make_data()
    model = FactorAnalysis(n_factors=2, max_iter=200, random_state=0).fit(x)

    z_hat = model.transform(x)
    recon = model.reconstruct(x)
    mse = float(np.mean((x - recon) ** 2))
    ll = model.log_likelihood(x)

    # Loadings are only identified up to rotation/sign, so compare the
    # implied covariance W W^T + Psi against the truth instead.
    cov_hat = model.components_ @ model.components_.T + np.diag(model.noise_variance_)
    cov_true = TRUE_W @ TRUE_W.T + np.diag(TRUE_PSI)
    cov_err = float(np.mean((cov_hat - cov_true) ** 2))

    print(f"Samples: {N_SAMPLES}, features: 6, factors: 2, EM iterations: {model.n_iter_}")
    print("True loadings (6 x 2):")
    print(np.round(TRUE_W, 3))
    print("Recovered loadings (up to rotation/sign):")
    print(np.round(model.components_, 3))
    print(f"True noise variance:     {np.round(TRUE_PSI, 3).tolist()}")
    print(f"Recovered noise variance: {np.round(model.noise_variance_, 3).tolist()}")
    print(f"Recovered mean:           {np.round(model.mean_, 3).tolist()}")
    print(f"Factor-score shape: {z_hat.shape}")
    print(f"Reconstruction MSE: {mse:.4f}")
    print(f"Covariance MSE vs truth: {cov_err:.4f}")
    print(f"Final log-likelihood: {ll:.2f}")

    assert z_hat.shape == (N_SAMPLES, 2)
    assert np.all(model.noise_variance_ > 0)
    assert mse < 0.3, f"reconstruction MSE too high: {mse:.4f}"
    assert cov_err < 0.05, f"covariance recovery too poor: {cov_err:.4f}"
    assert model.log_likelihood_history_[-1] >= model.log_likelihood_history_[0]
    print("All demo checks passed.")


if __name__ == "__main__":
    main()
