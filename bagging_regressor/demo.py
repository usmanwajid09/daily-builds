"""Fits a single decision stump and a bagged ensemble of 50 stumps on noisy sine data, then compares test MSE and R²."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bagging_regressor.model import BaggingRegressor, DecisionStumpRegressor


def make_data(seed: int = 42, n: int = 200, noise: float = 0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, size=n)
    y = np.sin(x) + rng.normal(0, noise, size=n)
    return x.reshape(-1, 1), y


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    ss_res = float(np.sum((y_true - np.asarray(y_pred)) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot


def main():
    X, y = make_data()
    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    stump = DecisionStumpRegressor().fit(X_train, y_train)
    bagged = BaggingRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)

    stump_mse = mse(y_test, stump.predict(X_test))
    bagged_mse = mse(y_test, bagged.predict(X_test))
    bagged_r2 = r2_score(y_test, bagged.predict(X_test))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"n_estimators=50")
    print(f"Single stump test MSE: {stump_mse:.4f}")
    print(f"Bagged (50 stumps) test MSE: {bagged_mse:.4f}")
    print(f"Bagged R²: {bagged_r2:.4f}")
    print(f"Bagging reduces test MSE by "
          f"{100 * (stump_mse - bagged_mse) / stump_mse:.1f}%")
    print(f"Single stump split: feature {stump.feature_index_}, "
          f"threshold {stump.threshold_:.3f}, "
          f"left {stump.left_value_:.3f}, right {stump.right_value_:.3f}")


if __name__ == "__main__":
    main()
