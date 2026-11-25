"""Demo: LOF anomaly detection on a dense blob with planted outliers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from local_outlier_factor.model import LocalOutlierFactor


def main():
    rng = np.random.default_rng(42)
    # Dense Gaussian blob (inliers) + a few far-away planted outliers.
    X_in = rng.normal(loc=0.0, scale=0.5, size=(100, 2))
    X_out = np.array([[8.0, 8.0], [-8.0, 7.0], [7.0, -8.0],
                      [-7.0, -7.0], [10.0, 0.0]])
    X = np.vstack([X_in, X_out])

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    labels = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_  # -LOF, higher = more normal

    flagged = labels[-5:]  # the 5 planted outliers
    n_flagged = int((flagged == -1).sum())
    inlier_lof = -scores[:100]
    outlier_lof = -scores[100:]

    print(f"Planted outliers flagged as -1: {n_flagged} / 5")
    print(f"Max LOF of planted outliers: {outlier_lof.max():.2f}")
    print(f"Median LOF of inliers:       {np.median(inlier_lof):.2f}")
    print(f"Max LOF of inliers:          {inlier_lof.max():.2f}")
    print(f"Total points flagged -1:     {int((labels == -1).sum())} / {len(X)}")
    print(f"Decision threshold (score):  {lof.threshold_:.3f}")


if __name__ == "__main__":
    main()
