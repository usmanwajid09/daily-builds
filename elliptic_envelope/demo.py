"""Generates clean 2D Gaussian data plus far outliers, fits an elliptic envelope, reports detection rates."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elliptic_envelope.model import EllipticEnvelope


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(200, 2))
    outliers = rng.normal(loc=[8.0, 8.0], scale=1.0, size=(10, 2))
    x = np.vstack([inliers, outliers])
    y = np.array([1] * 200 + [-1] * 10)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    model = EllipticEnvelope(contamination=0.05, random_state=42)
    model.fit(x)

    preds = model.predict(x)
    true_out = y == -1
    true_in = y == 1
    outlier_recall = float(np.mean(preds[true_out] == -1))
    inlier_accuracy = float(np.mean(preds[true_in] == 1))
    print(f"Outliers injected:   {int(true_out.sum())}")
    print(f"Outliers detected:   {int((preds[true_out] == -1).sum())} "
          f"(recall {outlier_recall:.2f})")
    print(f"Inliers kept:        {int((preds[true_in] == 1).sum())}/{int(true_in.sum())} "
          f"(accuracy {inlier_accuracy:.2f})")
    print(f"Total flagged:       {int((preds == -1).sum())}")
    print(f"Threshold (chi2):    {model.threshold_:.2f}")
    print(f"Robust centre:       {np.round(model.location_, 3).tolist()}")
    print("First 5 predictions:", preds[:5].tolist())
    print("First 5 true labels: ", y[:5].tolist())


if __name__ == "__main__":
    main()
