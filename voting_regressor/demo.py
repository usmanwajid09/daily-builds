"""Fits VotingRegressor (mean, median and weighted voting) and a single Ridge baseline on synthetic sine + quadratic data, then compares test RMSE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from voting_regressor.model import RidgeRegressor, VotingRegressor


def make_data(seed=42, n=200, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, noise, size=n)
    return X, y


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main():
    X, y = make_data()
    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    ridge = RidgeRegressor(alpha=1.0).fit(X_train, y_train)
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Single Ridge baseline test RMSE: {rmse(y_test, ridge.predict(X_test)):.4f}")

    for label, kwargs in [
        ("mean", {"voting": "mean"}),
        ("median", {"voting": "median"}),
        ("weighted [0.5, 0.3, 0.2]", {"voting": "weighted", "weights": [0.5, 0.3, 0.2]}),
    ]:
        model = VotingRegressor(**kwargs).fit(X_train, y_train)
        print(f"VotingRegressor ({label}) test RMSE: "
              f"{rmse(y_test, model.predict(X_test)):.4f}")

    # Show the three base learners disagreeing, and how voting reconciles them.
    mean_model = VotingRegressor(voting="mean").fit(X_train, y_train)
    base = np.column_stack(
        [est.predict(X_test[:5]) for est in mean_model.estimators_]
    )
    print("First 5 test samples — base predictions [Ridge, k-NN, Tree] vs mean vote:")
    for i in range(5):
        print(f"  true={y_test[i]:+.3f}  base={np.round(base[i], 3)}  "
              f"mean={mean_model.predict(X_test[i:i + 1])[0]:+.3f}")


if __name__ == "__main__":
    main()
