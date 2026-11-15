"""Generates three synthetic 2D Gaussian clusters, fits LDA, and
reports test accuracy plus the confusion between classes."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lda_classifier.model import LDAClassifier


def make_data(seed: int = 42, n: int = 60):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [4.0, 3.0], [0.0, 5.0]])
    xs, ys = [], []
    for k, c in enumerate(centres):
        xs.append(rng.normal(loc=c, scale=1.0, size=(n, 2)))
        ys.append(np.full(n, k))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = LDAClassifier().fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    probs = model.predict_proba(x_test[:3])

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, classes: 3")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Class priors: {np.round(model.priors_, 3).tolist()}")
    print(f"First 3 predicted probabilities:\n{np.round(probs, 3).tolist()}")
    print(f"First 5 predictions: {model.predict(x_test[:5]).tolist()}")


if __name__ == "__main__":
    main()
