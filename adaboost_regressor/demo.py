"""Demo: AdaBoost regression on Friedman #1 synthetic data.

y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise — a
non-linear, multi-feature target where a single stump is weak but the
boosted ensemble of stumps steadily improves. Prints the test R^2 of
the full ensemble and of the first 1, 10 and 50 stumps, plus the feature
importances.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from adaboost_regressor.model import AdaBoostRegressor


def make_data(seed: int = 42, n: int = 300):
    """Friedman #1 regression data: 5 uniform features, noisy target."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 5))
    y = (10 * np.sin(np.pi * x[:, 0] * x[:, 1])
         + 20 * (x[:, 2] - 0.5) ** 2
         + 10 * x[:, 3] + 5 * x[:, 4]
         + rng.normal(scale=0.5, size=n))
    return x, y


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = AdaBoostRegressor(n_estimators=100, learning_rate=1.0)
    model.fit(x_train, y_train)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Test R^2 (100 stumps): {model.score(x_test, y_test):.3f}")
    staged = list(model.staged_predict(x_test))
    for k in (1, 10, 50):
        print(f"  first {k:>3} stumps: test R^2 = "
              f"{_r2(y_test, staged[k - 1]):.3f}")
    print(f"Estimators used: {len(model.estimators_)}")
    print("Feature importances:",
          np.round(model.feature_importances_, 3).tolist())


if __name__ == "__main__":
    main()
