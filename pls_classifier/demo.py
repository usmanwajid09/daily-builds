"""Demo: PLS-DA on collinear synthetic data.

Features are built from just 2 latent factors plus noise (10 features,
strongly collinear), with 3 classes defined by the latent factors.
Shows accuracy for 1, 2 and 3 components.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pls_classifier.model import PLSClassifier


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 60
    centers = [[-3.0, -3.0], [3.0, -3.0], [0.0, 3.0]]
    xs, ys = [], []
    for i, c in enumerate(centers):
        latent = rng.normal(loc=c, scale=1.0, size=(n, 2))
        mixing = rng.normal(size=(2, 10))
        x = latent @ mixing + rng.normal(scale=0.3, size=(n, 10))
        xs.append(x)
        ys.append(np.full(n, i))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    for n_comp in (1, 2, 3):
        model = PLSClassifier(n_components=n_comp).fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(
            f"n_components={n_comp}  test accuracy: {acc:.3f}  "
            f"X variance explained: {model.explained_variance_ratio_.sum():.3f}"
        )

    model = PLSClassifier(n_components=2).fit(x_train, y_train)
    print(f"classes: {model.classes_.tolist()}")
    print(f"coef shape: {model.coef_.shape}")


if __name__ == "__main__":
    main()
