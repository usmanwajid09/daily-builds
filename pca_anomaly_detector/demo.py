"""Demo: flag outliers in correlated 2D data with PCA reconstruction error."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from pca_anomaly_detector.model import PCAAnomalyDetector


def main():
    rng = np.random.default_rng(0)
    n = 200
    factor = rng.normal(size=n)
    normal = np.column_stack([
        3.0 * factor + rng.normal(scale=0.2, size=n),
        3.0 * factor + rng.normal(scale=0.2, size=n),
    ])
    outliers = np.array([[8.0, -8.0], [-8.0, 8.0], [0.0, 8.0], [8.0, 0.0], [-8.0, 0.0]])
    rng_test = np.random.default_rng(1)
    factor_test = rng_test.normal(size=20)
    normal_test = np.column_stack([
        3.0 * factor_test + rng_test.normal(scale=0.2, size=20),
        3.0 * factor_test + rng_test.normal(scale=0.2, size=20),
    ])
    test = np.vstack([normal_test, outliers])

    det = PCAAnomalyDetector(n_components=1, threshold="quantile", quantile=0.99).fit(normal)
    errors = det.score_samples(test)
    flagged = det.is_anomaly(test)
    print(f"Training samples: {n}, components kept: 1, threshold: {det.threshold_:.4f}")
    print(f"Mean train reconstruction error: {det.train_errors_.mean():.4f}")
    print(f"Test set: 20 normal + 5 outliers; flagged as anomalies: {int(flagged.sum())}")
    print(f"Outliers flagged: {int(flagged[20:].sum())} / 5")
    print(f"Normal points flagged: {int(flagged[:20].sum())} / 20 (false positives)")
    print(f"Max normal-point error: {errors[:20].max():.4f}, min outlier error: {errors[20:].min():.4f}")


if __name__ == "__main__":
    main()
