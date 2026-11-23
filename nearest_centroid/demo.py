"""Demo: nearest-centroid classifier on synthetic blobs.

Shows test accuracy for shrinkage 0.0, 0.3 and 0.7, and for the
Manhattan metric, on three 2D clusters.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nearest_centroid.model import NearestCentroid


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

    for shrinkage in (0.0, 0.3, 0.7):
        model = NearestCentroid(shrinkage=shrinkage).fit(x_train, y_train)
        print(
            f"shrinkage={shrinkage:.1f}  test accuracy: "
            f"{model.score(x_test, y_test):.3f}"
        )

    model = NearestCentroid(metric="manhattan").fit(x_train, y_train)
    print(f"manhattan metric  test accuracy: {model.score(x_test, y_test):.3f}")

    model = NearestCentroid().fit(x_train, y_train)
    print(f"centroids:\n{model.centroids_.round(2)}")
    print(f"first test probabilities: {model.predict_proba(x_test[:1])[0].round(3).tolist()}")


if __name__ == "__main__":
    main()
