"""Demo: ridge classifier on two synthetic 2D clusters plus a 3-class set."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ridge_classifier.model import RidgeClassifier


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 60
    centers = [[0.0, 0.0], [4.0, 3.0], [0.0, 5.0]]
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(rng.normal(loc=c, scale=1.0, size=(n, 2)))
        ys.append(np.full(n, i))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_blobs()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    for alpha in (0.1, 1.0, 10.0):
        model = RidgeClassifier(alpha=alpha).fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(f"alpha={alpha:<5} test accuracy: {acc:.3f}")

    model = RidgeClassifier(alpha=1.0).fit(x_train, y_train)
    print(f"classes: {model.classes_.tolist()}")
    print(f"coef shape: {model.coef_.shape}")
    print(f"first test decision values: {model.decision_function(x_test[:1])[0].round(3).tolist()}")


if __name__ == "__main__":
    main()
