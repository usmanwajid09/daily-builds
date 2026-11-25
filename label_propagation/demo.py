"""Demo: label propagation on two blobs with only 10 labels per class."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from label_propagation.model import LabelPropagation


def make_blobs(seed=0, n_per_class=100):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 5.0]])
    X = np.vstack([rng.normal(c, 0.8, size=(n_per_class, 2)) for c in centres])
    y_true = np.repeat([0, 1], n_per_class)
    return X, y_true


def main():
    X, y_true = make_blobs()
    rng = np.random.default_rng(1)
    y = np.full(len(X), -1)
    n_labelled_per_class = 10
    for cls in (0, 1):
        idx = np.where(y_true == cls)[0]
        y[rng.choice(idx, size=n_labelled_per_class, replace=False)] = cls

    model = LabelPropagation(n_neighbors=7, alpha=0.8)
    model.fit(X, y)
    pred = model.predict(X)

    n_labelled = int(np.sum(y != -1))
    n_unlabelled = int(np.sum(y == -1))
    unlabeled_mask = y == -1
    acc_unlabeled = float(np.mean(pred[unlabeled_mask] == y_true[unlabeled_mask]))
    acc_all = float(np.mean(pred == y_true))

    print(f"Label propagation on 2 blobs: {len(X)} points, "
          f"{n_labelled} labelled, {n_unlabelled} unlabeled")
    print(f"Accuracy on unlabeled points: {acc_unlabeled:.3f}")
    print(f"Accuracy on all points:       {acc_all:.3f}")
    print(f"Converged in {model.n_iter_} iterations; "
          f"label distributions rows sum to 1: "
          f"{np.allclose(model.label_distributions_.sum(axis=1), 1.0)}")


if __name__ == "__main__":
    main()
