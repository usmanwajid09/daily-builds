"""Demo: fit SVRRegressor on a noisy sine wave and report test performance."""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from svr_regressor.model import SVRRegressor


def main():
    rng = np.random.default_rng(42)
    n = 120
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, 0.05, size=n)

    # 80/20 split (shuffled, seeded for reproducibility).
    indices = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = SVRRegressor(epsilon=0.05, C=10.0, learning_rate=0.01,
                         n_epochs=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    mse = float(np.mean((y_test - y_pred) ** 2))

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Test R^2: {r2:.4f}")
    print(f"Test MSE: {mse:.4f}")
    assert r2 > 0.8, f"Test R^2 {r2:.4f} did not beat 0.8"


if __name__ == "__main__":
    main()
