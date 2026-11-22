"""Demo: one-class SVM anomaly detection on synthetic 2D data.

Fits on 200 points from a Gaussian blob, then scores 50 fresh inliers
and 20 far-away outliers. Prints the inlier rate on fresh normal data
and the detection rate on the outliers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from one_class_svm.model import OneClassSVM


def main():
    rng = np.random.default_rng(42)
    x_train = rng.normal(0, 1, size=(200, 2))
    x_normal = rng.normal(0, 1, size=(50, 2))
    x_outlier = rng.normal(0, 1, size=(20, 2)) + np.array([8.0, 8.0])

    model = OneClassSVM(nu=0.1, random_state=0)
    model.fit(x_train)
    print(f"Train size: {len(x_train)}, support vectors: {len(model.support_)}")
    print(f"rho: {model.rho_:.4f}, gamma: {model.gamma_:.4f}")
    inlier_rate = float(np.mean(model.predict(x_normal) == 1))
    outlier_rate = float(np.mean(model.predict(x_outlier) == -1))
    print(f"Fresh inliers classified as normal: {inlier_rate:.2f}")
    print(f"Outliers detected as anomalies: {outlier_rate:.2f}")
    print(f"Mean decision score (inliers): {model.decision_function(x_normal).mean():.3f}")
    print(f"Mean decision score (outliers): {model.decision_function(x_outlier).mean():.3f}")


if __name__ == "__main__":
    main()
