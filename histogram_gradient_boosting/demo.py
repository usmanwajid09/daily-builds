"""Demo: histogram gradient boosting on synthetic noisy sinusoidal data.

Fits 200 shallow trees (max_bins=32) on y = sin(x) + noise and prints
the test R^2 plus the training MSE of the first and last stages.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from histogram_gradient_boosting.model import HistogramGradientBoostingRegressor


def main():
    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, size=(400, 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.normal(0, 0.1, size=400)
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = HistogramGradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3,
        max_bins=32, min_samples_leaf=10, random_state=0)
    model.fit(x_train, y_train)
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Histogram GB test R^2: {model.score(x_test, y_test):.3f}")
    print(f"Train MSE after 1 tree: {model.train_score_[0]:.4f}")
    print(f"Train MSE after 200 trees: {model.train_score_[-1]:.4f}")


if __name__ == "__main__":
    main()
