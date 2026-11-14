"""Generates 200 normal points plus 10 planted outliers, fits an isolation
forest, and reports how many of the planted outliers were caught.
"""
import os
import sys

# Running as `python isolation_forest/demo.py` puts the package directory,
# not the repo root, on sys.path; add the repo root so the package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from isolation_forest.model import IsolationForest


def make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    # Clear outliers, far from the normal cloud.
    angles = rng.uniform(0, 2 * np.pi, size=10)
    radii = rng.uniform(8, 10, size=10)
    outliers = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    x = np.vstack([inliers, outliers])
    return x, np.array([1] * 200 + [-1] * 10)


def main():
    x, truth = make_data()
    model = IsolationForest(contamination=0.05, random_state=0).fit(x)
    scores = model.decision_function(x)
    preds = model.predict(x)
    n_outliers = int(np.sum(preds == -1))
    caught = int(np.sum(preds[truth == -1] == -1))
    mean_inlier = float(np.mean(scores[truth == 1]))
    mean_outlier = float(np.mean(scores[truth == -1]))

    print(f"Samples: {len(x)} (200 inliers + 10 planted outliers)")
    print(f"Outliers detected: {n_outliers}")
    print(f"Planted outliers caught: {caught} / 10")
    print(f"Mean score - inliers: {mean_inlier:.3f}, outliers: {mean_outlier:.3f}")
    print(f"Threshold: {model.threshold_:.3f}")


if __name__ == "__main__":
    main()
