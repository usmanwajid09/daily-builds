"""Generates two concentric circles, fits an RBF-kernel SVM, and reports accuracy."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from svm_rbf.model import RBFKernelSVM


def make_circles(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 100
    angle_inner = rng.uniform(0, 2 * np.pi, n)
    angle_outer = rng.uniform(0, 2 * np.pi, n)
    inner = np.column_stack(
        [np.cos(angle_inner), np.sin(angle_inner)]
    ) * 1.0 + rng.normal(0, 0.1, size=(n, 2))
    outer = np.column_stack(
        [np.cos(angle_outer), np.sin(angle_outer)]
    ) * 3.0 + rng.normal(0, 0.1, size=(n, 2))
    X = np.vstack([inner, outer])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def main():
    X, y = make_circles()
    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = RBFKernelSVM(C=1.0, random_state=42).fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, C=1.0, gamma='scale'")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Number of support vectors: {model.n_support_} "
          f"(of {len(y_train)} training points)")
    print(f"Intercept (bias): {model.intercept_:.3f}")


if __name__ == "__main__":
    main()
