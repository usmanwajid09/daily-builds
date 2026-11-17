"""Demo: Bayesian ridge regression on y = 2x - 1 + noise.

Fits on 80 training points and reports the learned slope/intercept
(expected near 2.0 / -1.0), test R^2 and MSE, and the mean predictive
standard deviation on the test points.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bayesian_ridge.model import BayesianRidgeRegressor

import numpy as np


def main():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.uniform(-3, 3, size=(n, 1))
    y = 2 * X[:, 0] - 1 + rng.normal(0, 0.5, size=n)

    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    model = BayesianRidgeRegressor(alpha=1.0, beta=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = float(np.mean((y_test - y_pred) ** 2))
    r2 = model.score(X_test, y_test)
    mean_std = float(model.predict_std(X_test).mean())

    print(f"Learned coefficient: {model.coef_[0]:.4f} (true: 2.0)")
    print(f"Learned intercept:   {model.intercept_:.4f} (true: -1.0)")
    print(f"Test R^2:            {r2:.4f}")
    print(f"Test MSE:            {mse:.4f}")
    print(f"Mean predictive std: {mean_std:.4f}")

    assert r2 > 0.9, f"test R^2 too low: {r2}"
    assert abs(model.coef_[0] - 2.0) < 0.2, f"coef off: {model.coef_[0]}"
    print("All checks passed.")


if __name__ == "__main__":
    main()
