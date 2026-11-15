"""Generates synthetic data where only 2 of 5 features matter (two of
them correlated), fits Elastic Net, and reports test R^2 plus the
fitted coefficients."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from elastic_net.model import ElasticNetRegression


def make_data(seed: int = 42, n: int = 200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    x[:, 1] = x[:, 0] + rng.normal(scale=0.3, size=n)  # correlated pair
    y = 2.0 * x[:, 0] - 1.5 * x[:, 2] + rng.normal(scale=0.5, size=n)
    return x, y


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = ElasticNetRegression(alpha=1.0, l1_ratio=0.5).fit(x_train, y_train)
    r2 = model.score(x_test, y_test)
    mse = float(np.mean((model.predict(x_test) - y_test) ** 2))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"alpha=1.0, l1_ratio=0.5")
    print(f"Test R^2: {r2:.3f}")
    print(f"Test MSE: {mse:.3f}")
    print(f"Coefficients: {np.round(model.coef_, 3).tolist()}")
    print(f"Intercept: {model.intercept_:.3f} (converged in {model.n_iter_} passes)")


if __name__ == "__main__":
    main()
