"""Generates three synthetic 2D clusters, fits a random forest, reports accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_forest.model import RandomForestClassifier


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 80
    centres = [[0.0, 0.0], [4.0, 3.0], [0.0, 5.0]]
    xs, ys = [], []
    for label, centre in enumerate(centres):
        xs.append(rng.normal(loc=centre, scale=0.8, size=(n, 2)))
        ys.extend([label] * n)
    x = np.vstack(xs)
    y = np.array(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_blobs()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = RandomForestClassifier(n_estimators=50, max_depth=6,
                                   random_state=42).fit(x_train, y_train)
    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, "
          f"n_estimators=50, max_depth=6")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Feature importances: {model.feature_importances_.round(3).tolist()}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
