"""Trains softmax regression from scratch on 3 well-separated 2D
Gaussian blobs (100 points per class) and reports train/test sizes,
epochs run, final loss, test accuracy, and the first 5 predictions
against their true labels."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from softmax_regression.model import SoftmaxRegression


def make_blobs(seed=42, n_per_class=100):
    """Three well-separated 2D Gaussian blobs, labels 0, 1, 2."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.5]])
    X = np.vstack([rng.normal(loc=c, scale=1.0, size=(n_per_class, 2))
                   for c in centres])
    y = np.repeat([0, 1, 2], n_per_class)
    return X, y


def main():
    X, y = make_blobs()
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = SoftmaxRegression(learning_rate=0.1, n_epochs=1000)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    accuracy = float(np.mean(test_pred == y_test))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Epochs run: {model.n_epochs_run_}")
    print(f"Final loss: {model.loss_history_[-1]:.4f}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("First 5 predictions vs true labels:")
    for pred, true in zip(test_pred[:5], y_test[:5]):
        print(f"  predicted {pred}, true {true}")
    assert accuracy >= 0.95, f"Test accuracy {accuracy:.3f} below 0.95"


if __name__ == "__main__":
    main()
