"""Fits a Gaussian process to a noisy sine wave and reports test R^2,
MSE, and the predictive uncertainty near vs far from the training data."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gaussian_process_regressor.model import GaussianProcessRegressor


def make_data(seed: int = 42, n: int = 40):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = np.sin(x[:, 0]) + rng.normal(scale=0.1, size=n)
    return x, y


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = GaussianProcessRegressor(length_scale=1.0, sigma_f=1.0,
                                     sigma_n=0.1).fit(x_train, y_train)
    mean, std = model.predict(x_test, return_std=True)
    r2 = model.score(x_test, y_test)
    mse = float(np.mean((mean - y_test) ** 2))
    # Uncertainty far outside the training range should be larger.
    _, std_far = model.predict(np.array([[10.0], [-10.0]]), return_std=True)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"length_scale=1.0, sigma_n=0.1")
    print(f"Test R^2: {r2:.3f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Mean predictive std on test points: {std.mean():.3f}")
    print(f"Predictive std far from data (x=±10): {np.round(std_far, 3).tolist()}")
    print(f"Log marginal likelihood: {model.log_marginal_likelihood():.2f}")


if __name__ == "__main__":
    main()
