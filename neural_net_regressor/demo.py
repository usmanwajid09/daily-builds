"""Fits the MLP regressor to a noisy sine curve and reports train/test R^2."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from neural_net_regressor.model import MLPRegressor


def make_data(seed: int = 42, n: int = 200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    y = np.sin(x) + 0.1 * rng.normal(size=n)
    return x.reshape(-1, 1), y


def main():
    X, y = make_data()
    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = MLPRegressor(
        hidden_units=32, learning_rate=0.01, epochs=500,
        batch_size=32, random_state=42,
    ).fit(X_train, y_train)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"hidden_units=32, epochs run: {model.n_epochs_} / 500")
    print(f"Train R^2: {model.score(X_train, y_train):.4f}")
    print(f"Test R^2:  {model.score(X_test, y_test):.4f}")
    print(f"Final train loss (MSE, standardized): {model.history_[-1]:.6f}")
    print("Predictions vs true values (first 5 test points):")
    preds = model.predict(X_test[:5])
    for xi, pred, true in zip(X_test[:5].ravel(), preds, y_test[:5]):
        print(f"  x={xi:6.3f}  predicted={pred:7.4f}  true={true:7.4f}")


if __name__ == "__main__":
    main()
