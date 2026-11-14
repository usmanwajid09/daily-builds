"""Fits KNN regression to noisy sin(x) and reports test MSE and R^2."""
import sys
from pathlib import Path

# Running as `python knn_regressor/demo.py` puts the script's own folder,
# not the repo root, on sys.path - add the repo root so the package import
# below resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from knn_regressor.model import KNNRegressor


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2 * np.pi, size=150)
    y = np.sin(x) + rng.normal(loc=0.0, scale=0.1, size=150)
    return x.reshape(-1, 1), y


def main():
    x, y = make_data()
    x_train, y_train = x[:100], y[:100]
    x_test, y_test = x[100:], y[100:]

    for weights in ("uniform", "distance"):
        model = KNNRegressor(k=5, weights=weights).fit(x_train, y_train)
        preds = model.predict(x_test)
        mse = float(np.mean((y_test - preds) ** 2))
        r2 = model.score(x_test, y_test)
        print(f"weights={weights}: test MSE = {mse:.4f}, test R^2 = {r2:.4f}")

    # One sample prediction vs its true value (k=5, distance weighting).
    model = KNNRegressor(k=5, weights="distance").fit(x_train, y_train)
    sample_x = x_test[0]
    sample_pred = float(model.predict(sample_x)[0])
    print(
        f"Sample: x = {sample_x[0]:.4f}, "
        f"predicted y = {sample_pred:.4f}, true y = {y_test[0]:.4f}"
    )


if __name__ == "__main__":
    main()
