"""Generates two overlapping 2D clusters, fits uniform and distance-weighted k-NN, compares accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weighted_knn.model import WeightedKNNClassifier


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 100
    class0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    class1 = rng.normal(loc=[2.5, 2.5], scale=1.0, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_blobs()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    for weights in ("uniform", "distance"):
        model = WeightedKNNClassifier(n_neighbors=5, weights=weights)
        model.fit(x_train, y_train)
        print(f"weights={weights!r:>10}: train accuracy {model.score(x_train, y_train):.3f}, "
              f"test accuracy {model.score(x_test, y_test):.3f}")

    model = WeightedKNNClassifier(n_neighbors=5, weights="distance").fit(x_train, y_train)
    print("First 5 test predictions:", model.predict(x_test[:5]).tolist())
    print("First 5 true labels:     ", y_test[:5].tolist())
    print("First test point probabilities:",
          np.round(model.predict_proba(x_test[:1])[0], 3).tolist())


if __name__ == "__main__":
    main()
