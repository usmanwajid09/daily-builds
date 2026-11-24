"""Demo: quantile regression on heteroscedastic data.

Generates ``y = 2x + noise`` where the noise spread grows with ``x``,
then fits the 0.1, 0.5 (median) and 0.9 conditional quantiles.  Because
the noise is heteroscedastic, the fitted quantile lines fan out: the
0.9 line sits well above the median line at large ``x``.  Prints each
quantile's slope/intercept, the empirical coverage of the 0.9 model
(fraction of points at or below its predictions) and the pinball loss
of the median model, with asserts on the quantities that must hold.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quantile_regressor.model import QuantileRegressor, pinball_loss


def make_data(seed: int = 0, n: int = 400):
    """Heteroscedastic data: y = 2x + noise, noise std grows with x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=(n, 1))
    noise = rng.normal(scale=0.2 + 0.3 * x[:, 0])
    y = 2.0 * x[:, 0] + noise
    return x, y


def main():
    x, y = make_data()
    print(f"n = {len(y)}, x in [{x.min():.2f}, {x.max():.2f}]")

    models = {}
    for q in (0.1, 0.5, 0.9):
        model = QuantileRegressor(
            quantile=q, max_iter=5000, learning_rate=0.05, random_state=0)
        model.fit(x, y)
        models[q] = model
        print(f"q={q:.1f}: slope={model.coef_[0]:.3f}, "
              f"intercept={model.intercept_:.3f}, "
              f"iterations={model.n_iter_}")

    coverage_90 = float(np.mean(y <= models[0.9].predict(x)))
    median_loss = pinball_loss(y, models[0.5].predict(x), 0.5)
    print(f"Coverage of q=0.9 model: {coverage_90:.3f} "
          f"(fraction of y <= prediction)")
    print(f"Pinball loss of median model: {median_loss:.3f}")
    print(f"Median model R^2: {models[0.5].score(x, y):.3f}")

    # The 0.9 quantile line must sit above the 0.1 line everywhere.
    assert np.all(models[0.9].predict(x) >= models[0.1].predict(x))
    # About 90% of points should fall below the 0.9 line (with slack
    # for finite-sample noise).
    assert 0.80 <= coverage_90 <= 0.98, f"coverage {coverage_90:.3f}"
    # The median line should recover the true slope of 2.
    assert abs(models[0.5].coef_[0] - 2.0) < 0.3, (
        f"median slope {models[0.5].coef_[0]:.3f} far from 2")
    # Median pinball loss must be well below the loss of a naive
    # constant (mean) prediction.
    naive = pinball_loss(y, np.full_like(y, y.mean()), 0.5)
    assert median_loss < naive, "median model worse than the mean"
    print("All demo asserts passed.")


if __name__ == "__main__":
    main()
