"""Compares a single decision tree against a bagged ensemble on two-moons data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bagging_classifier.model import BaggingClassifier, _DecisionTree


def make_moons(seed: int = 42, n: int = 200, noise: float = 0.3):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta = rng.uniform(0, np.pi, size=n_half)
    upper = np.column_stack([np.cos(theta), np.sin(theta)])
    lower = np.column_stack([1.0 - np.cos(theta), 0.5 - np.sin(theta)])
    x = np.vstack([upper, lower]) + rng.normal(0, noise, size=(n, 2))
    y = np.array([0] * n_half + [1] * n_half)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def main():
    x, y = make_moons()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    tree = _DecisionTree().fit(x_train, y_train)
    single_acc = accuracy(y_test, tree.predict(x_test))

    ensemble = BaggingClassifier(n_estimators=25, random_state=42)
    ensemble.fit(x_train, y_train)
    bagged_acc = accuracy(y_test, ensemble.predict(x_test))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Single tree accuracy:   {single_acc:.3f}")
    print(f"Bagged (25 trees) acc:  {bagged_acc:.3f}")
    print(f"Improvement:            {bagged_acc - single_acc:+.3f}")
    print(f"First 5 predictions: {ensemble.predict(x_test)[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
