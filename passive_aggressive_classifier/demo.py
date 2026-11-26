"""Trains a Passive-Aggressive classifier online on a separable stream, reports accuracy and updates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from passive_aggressive_classifier.model import PassiveAggressiveClassifier


def make_stream(seed: int = 42, n: int = 300):
    rng = np.random.default_rng(seed)
    # Linearly separable: label is the sign of x0 + x1.
    x = rng.normal(0, 1, size=(n, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y


def main():
    x, y = make_stream()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    # Online training: feed the stream one example at a time.
    model = PassiveAggressiveClassifier(C=1.0)
    model.partial_fit(x_train[:1], y_train[:1], classes=np.array([0, 1]))
    for xi, yi in zip(x_train[1:], y_train[1:]):
        model.partial_fit(xi.reshape(1, -1), np.array([yi]))

    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Examples seen: {model.n_seen_}, weight updates: {model.n_updates_}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Weights: {np.round(model.coef_, 3).tolist()}, intercept: {model.intercept_:.3f}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
