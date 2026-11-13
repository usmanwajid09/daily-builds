"""Demo: perceptron on synthetic linearly separable data.

Generates two Gaussian blobs separated by a line, fits a perceptron,
and prints the test accuracy plus how many epochs it took to converge.
The true boundary is the line x0 + x1 = 2.
"""

import numpy as np

from perceptron.model import Perceptron


def make_data(seed: int = 42, n: int = 100):
    """Return (x, y): blobs centred at (0, 0) and (3, 3)."""
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=[0.0, 0.0], scale=0.5, size=(n, 2))
    class1 = rng.normal(loc=[3.0, 3.0], scale=0.5, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = Perceptron(max_epochs=100, random_state=42).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Epochs used: {model.n_epochs_} of {model.max_epochs}")
    print(f"Training errors of final weights: {model.n_errors_}")
    print(f"Learned weights: {np.round(model.coef_, 3).tolist()}, "
          f"bias: {model.intercept_:.3f}")


if __name__ == "__main__":
    main()
