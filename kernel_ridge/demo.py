"""Fits kernel ridge regression to a noisy sine wave — a non-linear
target a plain linear model cannot fit — and reports test R^2 and MSE."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from kernel_ridge.model import KernelRidgeRegressor


def make_data(seed: int = 42, n: int = 100):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = np.sin(x[:, 0]) + rng.normal(scale=0.1, size=n)
    return x, y


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = KernelRidgeRegressor(alpha=0.1, gamma=0.5).fit(x_train, y_train)
    r2 = model.score(x_test, y_test)
    mse = float(np.mean((model.predict(x_test) - y_test) ** 2))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"alpha=0.1, gamma=0.5")
    print(f"Test R^2: {r2:.3f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"First 5 predictions: {np.round(model.predict(x_test[:5]), 3).tolist()}")
    print(f"First 5 true values: {np.round(y_test[:5], 3).tolist()}")


if __name__ == "__main__":
    main()
