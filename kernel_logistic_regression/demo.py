"""Demo: kernel logistic regression on two moons.

The two-moons shape is not linearly separable, so a linear model fails
while the RBF kernel succeeds. Prints test accuracy for a linear-scale
gamma and a well-chosen gamma.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_logistic_regression.model import KernelLogisticRegression


def make_moons(seed: int = 42, n: int = 100):
    rng = np.random.default_rng(seed)
    t0 = rng.uniform(0, np.pi, n)
    t1 = rng.uniform(0, np.pi, n)
    x0 = np.column_stack([np.cos(t0), np.sin(t0)])
    x1 = np.column_stack([1.0 - np.cos(t1), 1.0 - np.sin(t1) - 0.5])
    x = np.vstack([x0, x1]) + rng.normal(scale=0.1, size=(2 * n, 2))
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_moons()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    for gamma in (0.5, 2.0, 8.0):
        model = KernelLogisticRegression(gamma=gamma, C=1.0).fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(
            f"gamma={gamma:<4} test accuracy: {acc:.3f}  "
            f"iters: {model.n_iter_}  "
            f"|dual coef| max: {np.abs(model.dual_coef_).max():.3f}"
        )

    model = KernelLogisticRegression(gamma=2.0, C=1.0).fit(x_train, y_train)
    probs = model.predict_proba(x_test[:3])
    print(f"first test probabilities: {probs.round(3).tolist()}")


if __name__ == "__main__":
    main()
