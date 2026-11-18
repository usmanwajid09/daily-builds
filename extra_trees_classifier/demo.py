"""Demo: extra trees classifier on synthetic two-moons-like data.

Fits a 100-tree forest with random split thresholds on two interleaved
crescents, and compares it against a single extra tree. Prints test
accuracy for both and the forest's feature importances.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from extra_trees_classifier.model import ExtraTreesClassifier


def make_moons(seed: int = 42, n: int = 200):
    """Two interleaved half-circles, like sklearn's make_moons."""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    t0 = rng.uniform(0, np.pi, n_half)
    t1 = rng.uniform(0, np.pi, n_half)
    x0 = np.column_stack([np.cos(t0), np.sin(t0)])
    x1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])
    x = np.vstack([x0, x1]) + rng.normal(scale=0.1, size=(n, 2))
    y = np.array([0] * n_half + [1] * n_half)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def main():
    x, y = make_moons()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    forest = ExtraTreesClassifier(n_estimators=100, random_state=0)
    forest.fit(x_train, y_train)
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Extra trees (100 trees) test accuracy: "
          f"{forest.score(x_test, y_test):.3f}")

    single = ExtraTreesClassifier(n_estimators=1, random_state=0)
    single.fit(x_train, y_train)
    print(f"Single extra tree test accuracy: "
          f"{single.score(x_test, y_test):.3f}")
    print("Feature importances:",
          np.round(forest.feature_importances_, 3).tolist())


if __name__ == "__main__":
    main()
