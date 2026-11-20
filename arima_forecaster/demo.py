"""Demo: fit an AR(2) forecaster to a synthetic series and forecast 5 steps."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from arima_forecaster.model import ARForecaster


def make_ar2(seed: int = 42, n: int = 500):
    """Synthetic AR(2): y[t] = 0.6*y[t-1] - 0.2*y[t-2] + noise."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0], y[1] = rng.normal(scale=1.0, size=2)
    noise = rng.normal(scale=1.0, size=n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + noise[t]
    return y


def main():
    y = make_ar2()
    model = ARForecaster(p=2).fit(y)
    print(f"True AR(2) coefficients: phi = [0.6, -0.2], intercept = 0")
    print(f"Fitted coefficients:     phi = {np.round(model.coef_, 4).tolist()}, "
          f"intercept = {model.intercept_:.4f}")
    print(f"Residual variance sigma2 = {model.sigma2_:.4f}")
    print(f"Last observed value: {y[-1]:.4f}")

    mean, lower, upper = model.forecast_interval(steps=5, alpha=0.05)
    print("\n5-step forecast (95% prediction interval):")
    for h in range(5):
        width = upper[h] - lower[h]
        print(f"  step {h + 1}: forecast = {mean[h]:+.4f}  "
              f"[{lower[h]:+.4f}, {upper[h]:+.4f}]  width = {width:.4f}")
    print("\nInterval width grows with horizon, as expected for an AR process.")


if __name__ == "__main__":
    main()
