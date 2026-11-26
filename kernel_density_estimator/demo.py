"""Fits a Gaussian KDE on a bimodal 1D mixture, reports densities and the estimated mode."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from kernel_density_estimator.model import GaussianKDE


def make_mixture(seed: int = 42, n: int = 300):
    rng = np.random.default_rng(seed)
    half = n // 2
    left = rng.normal(loc=-2.0, scale=0.8, size=half)
    right = rng.normal(loc=2.5, scale=1.0, size=n - half)
    return np.concatenate([left, right])


def main():
    samples = make_mixture()
    kde = GaussianKDE().fit(samples)
    points = np.array([-2.0, 0.0, 2.5])
    densities = kde.score(points)
    log_densities = kde.score_samples(points)
    mode = kde.mode()
    new_samples = kde.sample(5, random_state=0)

    print(f"Samples: {len(samples)}, bandwidth (Silverman): {kde.bandwidth_:.4f}")
    for point, density, log_density in zip(points, densities, log_densities):
        print(f"  x={point:5.1f}  density={density:.4f}  log-density={log_density:.4f}")
    print(f"Estimated mode: {mode[0]:.3f}")
    print(f"New samples: {np.round(new_samples.ravel(), 3).tolist()}")


if __name__ == "__main__":
    main()
