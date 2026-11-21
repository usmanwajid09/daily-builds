"""Demo: PLS regression on synthetic collinear data.

Generates 100 samples with 10 features built from just 3 latent
factors (so the features are strongly collinear — a case where
ordinary least squares is unstable), then fits PLS with 1, 2 and
3 components and prints the training R^2 for each, plus the
coefficients of the 3-component model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pls_regression.model import PLSRegression


def make_data(seed: int = 42, n: int = 100):
    """Return (X, y) built from 3 latent factors plus noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))  # the true drivers
    loadings = rng.normal(size=(3, 10))
    X = latent @ loadings + rng.normal(scale=0.1, size=(n, 10))
    beta = np.array([2.0, -1.5, 0.5])
    y = latent @ beta + rng.normal(scale=0.5, size=n)
    return X, y


def main():
    X, y = make_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features "
          f"(built from 3 latent factors, so highly collinear)")

    for n_comp in (1, 2, 3):
        model = PLSRegression(n_components=n_comp).fit(X, y)
        print(f"n_components={n_comp}: train R^2 = {model.score(X, y):.4f}")

    model = PLSRegression(n_components=3).fit(X, y)
    print(f"coef_ shape: {model.coef_.shape}")
    print("First 5 coefficients:", np.round(model.coef_[:5, 0], 3).tolist())
    print(f"intercept: {model.intercept_[0]:.3f}")
    print(f"X-scores shape (transform): {model.transform(X).shape}")


if __name__ == "__main__":
    main()
