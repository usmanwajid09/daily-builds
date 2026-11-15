"""Generates two interleaving moons (a non-linear boundary), trains a
small MLP, and reports test accuracy plus final training loss."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mlp_classifier.model import MLPClassifier


def make_moons(seed: int = 42, n: int = 100):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, np.pi, size=n)
    outer = np.c_[np.cos(t), np.sin(t)]
    inner = np.c_[1.0 - np.cos(t), 0.5 - np.sin(t)]
    x = np.vstack([outer, inner]) + rng.normal(scale=0.1, size=(2 * n, 2))
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_moons()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = MLPClassifier(hidden_size=16, learning_rate=0.1, epochs=300,
                          batch_size=32, random_state=42).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"hidden_size=16, epochs=300")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Final training loss: {model.loss_history_[-1]:.4f}")
    print(f"First 5 predictions: {model.predict(x_test[:5]).tolist()}")


if __name__ == "__main__":
    main()
