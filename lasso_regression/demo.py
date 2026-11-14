"""Generates synthetic regression data, fits Lasso, reports R^2 and coefficients."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lasso_regression.model import LassoRegression


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.normal(size=(n, 5))
    # Only the first two features matter; the rest are pure noise.
    y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(scale=0.5, size=n)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = LassoRegression(alpha=5.0).fit(x_train, y_train)
    r2 = model.score(x_test, y_test)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, alpha=5.0")
    print(f"Test R^2: {r2:.3f}")
    print(f"Coefficients: {model.coef_.round(3).tolist()}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"Non-zero coefficients: {int(np.sum(model.coef_ != 0))} of {len(model.coef_)}")
    print(f"Converged in {model.n_iter_} coordinate-descent passes")


if __name__ == "__main__":
    main()
