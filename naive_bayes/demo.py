"""Generates synthetic Gaussian classes, fits Gaussian NB, reports accuracy."""
import numpy as np

from naive_bayes.model import GaussianNB


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 60
    class0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    class1 = rng.normal(loc=[3.0, 3.0], scale=1.5, size=(n, 2))
    class2 = rng.normal(loc=[0.0, 4.0], scale=0.8, size=(n, 2))
    x = np.vstack([class0, class1, class2])
    y = np.array([0] * n + [1] * n + [2] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = GaussianNB().fit(x_train, y_train)
    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, classes: 3")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Class priors: {np.round(np.exp(model._log_prior), 3).tolist()}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
