"""Generates two synthetic 2D clusters, fits a stacking ensemble, reports accuracy."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stacking_ensemble.model import (
    DecisionStump,
    SimpleLogisticRegression,
    StackingClassifier,
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

    base = [SimpleLogisticRegression(lr=0.1, n_iter=500), DecisionStump()]
    meta = SimpleLogisticRegression(lr=0.1, n_iter=500)
    model = StackingClassifier(base, meta, n_folds=5).fit(x_train, y_train)
    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))

    # Compare against each base learner alone.
    base_accs = []
    for est in model.base_estimators_:
        base_accs.append(float(np.mean(est.predict(x_test) == y_test)))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Base learners: SimpleLogisticRegression, DecisionStump")
    print(f"Base learner test accuracies: {[round(a, 3) for a in base_accs]}")
    print(f"Stacking test accuracy: {accuracy:.3f}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
