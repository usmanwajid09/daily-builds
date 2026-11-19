"""Generates two synthetic 2D clusters, fits hard and soft voting ensembles, reports accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voting_classifier.model import (
    SimpleKNNClassifier,
    SimpleLogisticRegression,
    VotingClassifier,
)


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

    estimators = [SimpleLogisticRegression(lr=0.1, n_iter=500), SimpleKNNClassifier(k=5)]
    soft = VotingClassifier(estimators, voting="soft").fit(x_train, y_train)
    hard = VotingClassifier(estimators, voting="hard").fit(x_train, y_train)
    soft_acc = float(np.mean(soft.predict(x_test) == y_test))
    hard_acc = float(np.mean(hard.predict(x_test) == y_test))
    base_accs = [
        float(np.mean(est.predict(x_test) == y_test)) for est in soft.estimators_
    ]

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Base learners: SimpleLogisticRegression, SimpleKNNClassifier(k=5)")
    print(f"Base learner test accuracies: {[round(a, 3) for a in base_accs]}")
    print(f"Soft voting test accuracy: {soft_acc:.3f}")
    print(f"Hard voting test accuracy: {hard_acc:.3f}")
    print(f"Soft voting proba (first 3 rows): {np.round(soft.predict_proba(x_test[:3]), 3).tolist()}")


if __name__ == "__main__":
    main()
