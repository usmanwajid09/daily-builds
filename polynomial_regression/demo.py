"""Fit polynomial regression to noisy quadratic data and select the degree.

Generates y = 2 + 3x - x^2 + noise (seed 42, x in [-3, 3], 120 points),
splits 70/15/15 into train/val/test, tries degrees 1..5 on the validation
set, and reports the best degree plus the winning model's test MSE and R^2.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from polynomial_regression.model import PolynomialRegression, select_degree


def make_data(seed: int = 42, n: int = 120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = 2 + 3 * x - x**2 + rng.normal(scale=0.5, size=n)
    return x, y


def main():
    x, y = make_data()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    n_train, n_val = int(0.70 * len(y)), int(0.15 * len(y))
    train_idx, val_idx, test_idx = (
        idx[:n_train],
        idx[n_train:n_train + n_val],
        idx[n_train + n_val:],
    )
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    best_degree, best_mse, results = select_degree(
        x_train, y_train, x_val, y_val, degrees=range(1, 6), alpha=1.0
    )

    print(f"Train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print("Validation MSE by degree:")
    for degree, mse in results.items():
        print(f"  degree {degree}: MSE = {mse:.4f}")
    print(f"Best degree: {best_degree} (val MSE = {best_mse:.4f})")

    model = PolynomialRegression(degree=best_degree, alpha=1.0).fit(x_train, y_train)
    test_mse = float(np.mean((model.predict(x_test) - y_test) ** 2))
    test_r2 = model.score(x_test, y_test)
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R^2: {test_r2:.4f}")
    print(f"Fitted coefficients (intercept first): {np.round(model.coef_, 4)}")
    print("True coefficients: [2, 3, -1]")
    assert test_r2 >= 0.90, f"test R^2 {test_r2:.4f} below 0.90"


if __name__ == "__main__":
    main()
