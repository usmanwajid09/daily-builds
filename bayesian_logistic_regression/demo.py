"""Demo: Bayesian logistic regression on two synthetic Gaussian blobs.

Fits with a Laplace posterior, prints test accuracy, the MAP weights,
and predictive probabilities for a point near the data and one far
away (the far point's probability is pulled towards 0.5).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bayesian_logistic_regression.model import BayesianLogisticRegression


def main():
    rng = np.random.default_rng(42)
    x0 = rng.normal([-2, 0], 1.0, size=(100, 2))
    x1 = rng.normal([2, 0], 1.0, size=(100, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 100 + [1] * 100)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = BayesianLogisticRegression(alpha=1.0)
    model.fit(x_train, y_train)
    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Test accuracy: {model.score(x_test, y_test):.3f}")
    print(f"Newton iterations: {model.n_iter_}")
    print(f"MAP coefficients: {np.round(model.coef_, 3).tolist()}, "
          f"intercept: {model.intercept_:.3f}")
    near = np.array([[0.0, 0.0]])
    far = np.array([[0.0, 50.0]])
    print(f"P(y=1 | near point [0, 0]): {model.predict_proba(near)[0, 1]:.3f} "
          f"(MAP plug-in: {model.predict_proba(near, use_uncertainty=False)[0, 1]:.3f})")
    print(f"P(y=1 | far point [0, 50]): {model.predict_proba(far)[0, 1]:.3f} "
          f"(MAP plug-in: {model.predict_proba(far, use_uncertainty=False)[0, 1]:.3f})")


if __name__ == "__main__":
    main()
