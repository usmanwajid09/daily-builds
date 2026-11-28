"""Generates two interleaving moons, fits a histogram gradient boosting classifier, reports accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from histogram_gb_classifier.model import HistogramGBClassifier


def make_moons(seed: int = 42, n: int = 150, noise: float = 0.15):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta0 = rng.uniform(0, np.pi, n_half)
    theta1 = rng.uniform(0, np.pi, n_half)
    moon0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
    moon1 = np.column_stack([1.0 - np.cos(theta1), 0.5 - np.sin(theta1)])
    x = np.vstack([moon0, moon1]) + rng.normal(scale=noise, size=(n, 2))
    y = np.array([0] * n_half + [1] * n_half)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_moons()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = HistogramGBClassifier(n_estimators=80, learning_rate=0.1,
                                  max_depth=3, max_bins=16)
    model.fit(x_train, y_train)

    print(f"Train accuracy: {model.score(x_train, y_train):.3f}")
    print(f"Test accuracy:  {model.score(x_test, y_test):.3f}")
    print(f"Trees per class: {len(model.boosters_[0][0])}")
    print(f"Bins per feature: {model.max_bins}")
    print("First 5 test predictions:", model.predict(x_test[:5]).tolist())
    print("First 5 true labels:     ", y_test[:5].tolist())
    print("First test point probabilities:",
          np.round(model.predict_proba(x_test[:1])[0], 3).tolist())


if __name__ == "__main__":
    main()
