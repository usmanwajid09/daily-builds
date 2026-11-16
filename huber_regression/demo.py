"""Demo: Huber regression vs OLS on data with gross outliers.

Generates y = 2 + 3x + small noise, corrupts 10% of the targets with large
outliers (+/-15), fits both Huber regression (IRLS) and OLS on the corrupted
data, and compares them on clean, noiseless test points. Huber should win by
a wide margin because the outliers barely move its fit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from huber_regression import HuberRegression, ols_fit


def main():
    rng = np.random.default_rng(42)

    # --- Training data: clean line + 10% gross outliers ---
    x = rng.uniform(0, 10, 100)
    y_clean = 2 + 3 * x + rng.normal(0, 0.5, 100)
    y = y_clean.copy()
    outlier_idx = rng.choice(100, size=10, replace=False)
    y[outlier_idx] += rng.choice([-15.0, 15.0], size=10)
    X = x.reshape(-1, 1)

    # --- Fit both models on the corrupted data ---
    huber = HuberRegression(delta=1.35).fit(X, y)
    ols_coef, ols_intercept = ols_fit(X, y)

    # --- Evaluate on fresh, clean, noiseless test data ---
    x_test = np.linspace(0, 10, 200)
    X_test = x_test.reshape(-1, 1)
    y_test = 2 + 3 * x_test
    huber_mse = float(np.mean((huber.predict(X_test) - y_test) ** 2))
    ols_mse = float(np.mean((X_test @ ols_coef + ols_intercept - y_test) ** 2))

    print(f"True:            slope=3.000, intercept=2.000")
    print(f"Huber:           slope={huber.coef_[0]:.3f}, intercept={huber.intercept_:.3f}")
    print(f"OLS:             slope={ols_coef[0]:.3f}, intercept={ols_intercept:.3f}")
    print(f"Huber test MSE:  {huber_mse:.4f}")
    print(f"OLS test MSE:    {ols_mse:.4f}")
    print(f"IRLS iterations: {huber.n_iter_run_}")
    print(f"Outlier weights: min={huber.weights_[outlier_idx].min():.3f}, "
          f"inlier weights: min={np.delete(huber.weights_, outlier_idx).min():.3f}")
    print(f"Huber test MSE ({huber_mse:.4f}) is lower than OLS test MSE ({ols_mse:.4f})")

    assert huber_mse < ols_mse, (
        f"Expected Huber test MSE ({huber_mse:.4f}) to be lower than "
        f"OLS test MSE ({ols_mse:.4f})"
    )


if __name__ == "__main__":
    main()
