"""Generates synthetic regression data, fits gradient boosting, reports R^2."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_boosting.model import GradientBoostingRegressor


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.normal(size=(n, 3))
    # Non-linear target: interaction + sine, so trees earn their keep.
    y = 3.0 * x[:, 0] + 2.0 * x[:, 0] * x[:, 1] + np.sin(2.0 * x[:, 2]) \
        + rng.normal(scale=0.3, size=n)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                      max_depth=3).fit(x_train, y_train)
    preds = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"n_estimators=150, learning_rate=0.1, max_depth=3")
    print(f"Test R^2: {r2:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"First 5 predictions: {preds[:5].round(3).tolist()}")
    print(f"First 5 true values: {y_test[:5].round(3).tolist()}")


if __name__ == "__main__":
    main()
