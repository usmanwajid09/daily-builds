"""Demo: Theil-Sen robust regression vs ordinary least squares.

True line y = 2x + 1, Gaussian noise, ~10% gross outliers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from theil_sen_regressor import TheilSenRegressor


def main():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.uniform(0, 10, n)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, n)
    # ~10% gross outliers
    out = rng.choice(n, size=10, replace=False)
    y[out] += rng.choice([-1, 1], size=10) * rng.uniform(8, 15, 10)

    model = TheilSenRegressor().fit(x, y)
    ols_slope, ols_intercept = np.polyfit(x, y, 1)

    ts_err = abs(model.coef_[0] - 2.0)
    ols_err = abs(ols_slope - 2.0)
    print(f"True slope:      2.0000  (intercept 1.0)")
    print(f"Theil-Sen slope: {model.coef_[0]:.4f}  intercept {model.intercept_:.4f}"
          f"  (slope error {ts_err:.4f})")
    print(f"OLS slope:       {ols_slope:.4f}  intercept {ols_intercept:.4f}"
          f"  (slope error {ols_err:.4f})")
    print(f"Theil-Sen R^2:   {model.score(x, y):.4f}")
    print(f"HEADLINE: Theil-Sen slope error {ts_err:.4f} (< 0.2 required) "
          f"vs OLS slope error {ols_err:.4f} with 10% gross outliers")


if __name__ == "__main__":
    main()
