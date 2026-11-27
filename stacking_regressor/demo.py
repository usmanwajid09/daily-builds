"""Demo: stacking regressor vs a single ridge baseline on a non-linear target.

Synthetic data: y = sin(3*x0) + x1**2 + noise. A plain linear ridge model
cannot capture the sine curve or the quadratic term, so the stacker — whose
tree and k-NN base learners can — should win clearly.

Run from the repo root: python stacking_regressor/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stacking_regressor.model import RidgeRegressor, StackingRegressor


def main():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.1, size=n)

    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    stacker = StackingRegressor(n_folds=5, alpha=1.0, random_state=42)
    stacker.fit(X_train, y_train)
    stack_rmse = float(
        np.sqrt(np.mean((stacker.predict(X_test) - y_test) ** 2))
    )

    baseline = RidgeRegressor(alpha=1.0)
    baseline.fit(X_train, y_train)
    base_rmse = float(
        np.sqrt(np.mean((baseline.predict(X_test) - y_test) ** 2))
    )

    print("Stacking regressor demo (y = sin(3*x0) + x1^2 + noise)")
    print(f"  train: {X_train.shape[0]} samples, test: {X_test.shape[0]} samples")
    print(f"  StackingRegressor RMSE: {stack_rmse:.4f}")
    print(f"  RidgeRegressor  RMSE:   {base_rmse:.4f}")
    print(
        "  Meta-learner weights [intercept, ridge, knn, tree]: "
        f"{np.round(stacker.meta_model_, 4).tolist()}"
    )
    print(
        f"  Stacker beats ridge by {base_rmse - stack_rmse:.4f} RMSE "
        f"({100 * (base_rmse - stack_rmse) / base_rmse:.1f}% lower)"
    )


if __name__ == "__main__":
    main()
