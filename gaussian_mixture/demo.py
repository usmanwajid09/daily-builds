"""Generates two synthetic 2D clusters, fits a GMM, reports recovered means."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_mixture.model import GaussianMixture


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 150
    blob0 = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n, 2))
    blob1 = rng.normal(loc=[5.0, 4.0], scale=0.8, size=(n, 2))
    x = np.vstack([blob0, blob1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_blobs()

    model = GaussianMixture(n_components=2, random_state=42).fit(x)
    preds = model.predict(x)
    # Best of the two label assignments (cluster ids are arbitrary).
    acc_direct = float(np.mean(preds == y))
    accuracy = max(acc_direct, 1.0 - acc_direct)
    proba = model.predict_proba(x)

    print(f"Samples: {len(y)}, n_components=2, EM iterations: {model.n_iter_}")
    print(f"Recovered means:\n{np.round(model.means_, 3).tolist()}")
    print(f"Weights: {model.weights_.round(3).tolist()}")
    print(f"Log-likelihood: {model.log_likelihood_:.2f}")
    print(f"Mean max responsibility: {float(proba.max(axis=1).mean()):.3f}")
    print(f"Clustering accuracy (best label assignment): {accuracy:.3f}")


if __name__ == "__main__":
    main()
