"""Demo: elastic-net logistic classifier on synthetic blobs.

Shows how the L1 ratio controls sparsity: with a strong L1 penalty and
several pure-noise features, most coefficients are driven to exactly zero.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elastic_net_classifier.model import ElasticNetClassifier


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 80
    # Two informative features + four pure-noise features.
    x0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    x1 = rng.normal(loc=[3.0, 2.5], scale=1.0, size=(n, 2))
    noise0 = rng.normal(size=(n, 4))
    noise1 = rng.normal(size=(n, 4))
    x = np.vstack([np.hstack([x0, noise0]), np.hstack([x1, noise1])])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    for l1_ratio in (0.0, 0.5, 1.0):
        model = ElasticNetClassifier(alpha=0.1, l1_ratio=l1_ratio).fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        n_zero = int(np.sum(model.coef_ == 0.0))
        print(
            f"l1_ratio={l1_ratio:.1f}  test accuracy: {acc:.3f}  "
            f"zero coefficients: {n_zero}/{model.coef_.size}  "
            f"iters: {model.n_iter_.tolist()}"
        )

    model = ElasticNetClassifier(alpha=0.1, l1_ratio=1.0).fit(x_train, y_train)
    print(f"coefficients (l1_ratio=1.0): {model.coef_[1].round(3).tolist()}")


if __name__ == "__main__":
    main()
