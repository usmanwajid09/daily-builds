"""Demo: gradient boosting classifier on synthetic two-moons-like data.

Two interleaved crescents are hard for a single stump but easy for a
boosted ensemble. Prints test accuracy, the accuracy after each of the
first few stages, and the feature importances.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gradient_boosting_classifier.model import GradientBoostingClassifier


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

    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                       max_depth=3, random_state=0)
    model.fit(x_train, y_train)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Test accuracy: {model.score(x_test, y_test):.3f}")
    staged = list(model.staged_predict(x_test))
    for n in (1, 5, 10, 25, 50):
        acc = float(np.mean(staged[n - 1] == y_test))
        print(f"  after stage {n:>2}: accuracy = {acc:.3f}")
    print("Feature importances:",
          np.round(model.feature_importances_, 3).tolist())


if __name__ == "__main__":
    main()
