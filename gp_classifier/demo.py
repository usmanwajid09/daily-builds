"""Demo: Gaussian process classifier on two well-separated blobs.

Fits with a Laplace approximation, prints train/test accuracy and the
mean predicted probability of class 1 on each true class.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gp_classifier.model import GaussianProcessClassifier


def main():
    rng = np.random.default_rng(42)
    x0 = rng.normal([-3, 0], 1.0, size=(100, 2))
    x1 = rng.normal([3, 0], 1.0, size=(100, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 100 + [1] * 100)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = GaussianProcessClassifier(length_scale=1.0, signal_variance=1.0)
    model.fit(x_train, y_train)

    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    proba = model.predict_proba(x_test)
    mean_p_class0 = float(np.mean(proba[y_test == 0, 1]))
    mean_p_class1 = float(np.mean(proba[y_test == 1, 1]))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Newton iterations: {model.n_iter_}")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Mean P(y=1) on true class 0: {mean_p_class0:.3f}")
    print(f"Mean P(y=1) on true class 1: {mean_p_class1:.3f}")
    print(f"P(y=1 | class 0 centre [-3, 0]): "
          f"{model.predict_proba(np.array([[-3.0, 0.0]]))[0, 1]:.3f}")
    print(f"P(y=1 | class 1 centre [3, 0]): "
          f"{model.predict_proba(np.array([[3.0, 0.0]]))[0, 1]:.3f}")
    # Far from all training data the GP reverts towards 0.5 (growing variance).
    print(f"P(y=1 | far from all data [0, 10]): "
          f"{model.predict_proba(np.array([[0.0, 10.0]]))[0, 1]:.3f}")

    assert train_acc > 0.9, train_acc
    assert test_acc > 0.9, test_acc
    assert mean_p_class0 < 0.2, mean_p_class0
    assert mean_p_class1 > 0.8, mean_p_class1
    print("All demo asserts passed.")


if __name__ == "__main__":
    main()
