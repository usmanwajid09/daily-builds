"""Runs 5-fold CV of a nearest-centroid classifier and reports mean +/- std accuracy."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cross_validator.model import StratifiedKFold, cross_val_score


class NearestCentroid:
    """Tiny classifier used for the demo: predicts the nearest class mean."""

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([x[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, x):
        dists = np.linalg.norm(x[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(dists, axis=1)]


def make_blobs(seed: int = 42, n: int = 150):
    rng = np.random.default_rng(seed)
    per_class = n // 3
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
    x = np.vstack([
        rng.normal(loc=c, scale=0.8, size=(per_class, 2)) for c in centres
    ])
    y = np.array([c for c in range(3) for _ in range(per_class)])
    perm = rng.permutation(n)
    return x[perm], y[perm]


def main():
    x, y = make_blobs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(NearestCentroid(), x, y, cv=cv, scoring="accuracy")
    print(f"Samples: {len(y)}, classes: {len(np.unique(y))}, folds: 5 (stratified, shuffled)")
    print(f"Fold accuracies: {np.round(scores, 3).tolist()}")
    print(f"Mean accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")


if __name__ == "__main__":
    main()
