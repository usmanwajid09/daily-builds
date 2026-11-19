"""Generates noisy monotone data, fits isotonic regression, reports MSE before/after."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isotonic_regression.model import IsotonicRegression


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, size=100))
    # True monotone curve: sqrt, plus noise.
    y_true = np.sqrt(x)
    y = y_true + rng.normal(scale=0.3, size=len(x))
    return x, y, y_true


def main():
    x, y, y_true = make_data()
    model = IsotonicRegression(increasing=True).fit(x, y)
    fitted = model.predict(x)
    mse_raw = float(np.mean((y - y_true) ** 2))
    mse_iso = float(np.mean((fitted - y_true) ** 2))
    monotone = bool(np.all(np.diff(fitted) >= 0))

    print(f"Samples: {len(x)}, x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Fitted values monotone non-decreasing: {monotone}")
    print(f"MSE of noisy y vs true curve: {mse_raw:.4f}")
    print(f"MSE of isotonic fit vs true curve: {mse_iso:.4f}")
    print(f"R^2 of isotonic fit: {model.score(x, y):.3f}")
    print(f"First 5 fitted values: {np.round(fitted[:5], 3).tolist()}")
    print(f"Last 5 fitted values: {np.round(fitted[-5:], 3).tolist()}")


if __name__ == "__main__":
    main()
