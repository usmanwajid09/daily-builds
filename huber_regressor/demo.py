"""Demo: Huber regression vs plain OLS on data with ~10% gross outliers.

Generates y = 2x - 1 + noise (fixed seed), corrupts ~10% of the points
with large outliers, then fits HuberRegressor and plain OLS (numpy
lstsq) on the same training set and compares them on a clean test set.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huber_regressor.model import HuberRegressor

import numpy as np


def main():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.uniform(-3, 3, size=(n, 1))
    y = 2 * X[:, 0] - 1 + rng.normal(0, 0.5, size=n)

    # Corrupt ~10% of the points with gross outliers (+8 to +12 shift).
    n_outliers = int(0.10 * n)
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    y[outlier_idx] += rng.uniform(8, 12, size=n_outliers)

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # Clean test targets (no outliers) so MSE measures the true fit.
    y_test_clean = 2 * X_test[:, 0] - 1 + rng.normal(0, 0.5, size=len(X_test))

    huber = HuberRegressor(epsilon=1.35)
    huber.fit(X_train, y_train)
    huber_pred = huber.predict(X_test)
    huber_mse = float(np.mean((y_test_clean - huber_pred) ** 2))

    # Plain OLS for comparison.
    X_aug = np.column_stack([np.ones(len(X_train)), X_train])
    ols_params, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)
    ols_pred = ols_params[0] + X_test[:, 0] * ols_params[1]
    ols_mse = float(np.mean((y_test_clean - ols_pred) ** 2))

    print(f"True slope / intercept:     2.0000 / -1.0000")
    print(f"Huber slope / intercept:    {huber.coef_[0]:.4f} / {huber.intercept_:.4f}  (IRLS iterations: {huber.n_iter_})")
    print(f"OLS slope / intercept:      {ols_params[1]:.4f} / {ols_params[0]:.4f}")
    print(f"Huber test MSE:             {huber_mse:.4f}")
    print(f"OLS test MSE:               {ols_mse:.4f}")
    print(f"Huber slope error:          {abs(huber.coef_[0] - 2.0):.4f}")
    print(f"OLS slope error:            {abs(ols_params[1] - 2.0):.4f}")

    assert abs(huber.coef_[0] - 2.0) < abs(ols_params[1] - 2.0), \
        "Huber should be closer to the true slope than OLS on outlier data"
    assert huber_mse < ols_mse, "Huber should have lower test MSE than OLS"
    print("OK: Huber beats OLS on outlier-contaminated data.")


if __name__ == "__main__":
    main()
