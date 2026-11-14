"""Generates two synthetic 2D clusters, fits a linear SVM, reports accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svm_from_scratch.model import LinearSVC


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 100
    class0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    class1 = rng.normal(loc=[4.0, 3.0], scale=1.0, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_blobs()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = LinearSVC(C=1.0, learning_rate=0.01, n_epochs=500,
                      random_state=42).fit(x_train, y_train)
    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))
    margins = model.decision_function(x_test)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, C=1.0")
    print(f"Weights: {model.coef_.round(3).tolist()}, bias: {model.intercept_:.3f}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Mean |decision value| on test set: {float(np.mean(np.abs(margins))):.3f}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
