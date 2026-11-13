"""Demo: ridge regression on synthetic noisy linear data.

Generates a 2-feature linear relationship with noise, fits ridge
regression with a few different alphas, and prints the R^2 and the
coefficients for each. The true relationship is y = 3*x0 - 2*x1 + 5.
"""

import numpy as np

from ridge_regression.model import RidgeRegression


def make_data(seed: int = 42, n: int = 120):
    """Return (x, y) with y = 3*x0 - 2*x1 + 5 + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    noise = rng.normal(scale=0.5, size=n)
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 5.0 + noise
    return x, y


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print("True coefficients: [3.0, -2.0], intercept 5.0")
    for alpha in (0.0, 1.0, 10.0):
        model = RidgeRegression(alpha=alpha).fit(x_train, y_train)
        r2 = model.score(x_test, y_test)
        coef = np.round(model.coef_, 3).tolist()
        print(f"alpha={alpha:>4}: R^2 = {r2:.3f}, coef = {coef}, "
              f"intercept = {model.intercept_:.3f}")


if __name__ == "__main__":
    main()
