"""Generates two synthetic 2D clusters, fits an Adaline classifier, reports accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaline_classifier.model import AdalineClassifier


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 100
    class0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    class1 = rng.normal(loc=[3.5, 3.0], scale=1.0, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_blobs()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = AdalineClassifier(learning_rate=0.05, n_iter=500)
    model.fit(x_train, y_train)

    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Epochs run:     {model.n_iter_}")
    print(f"Final cost:     {model.costs_[0][-1]:.4f}")
    print(f"Coefficients:   {np.round(model.coef_[0], 3)}")
    print(f"Intercept:      {np.round(model.intercept_[0], 3)}")
    print("First 5 test predictions:", model.predict(x_test[:5]).tolist())
    print("First 5 true labels:     ", y_test[:5].tolist())


if __name__ == "__main__":
    main()
