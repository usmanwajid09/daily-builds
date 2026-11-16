"""Demo: QDA on 3 Gaussian classes with different covariances.

Each class has its own spread, so a shared-covariance (LDA-style) model
cannot separate them as well: class 0 is elongated along x, class 1 is
elongated along y, and class 2 is isotropic. QDA fits a covariance per
class, which is exactly what this data needs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from qda_classifier import QDAClassifier


def make_data(seed=42, n_per_class=100):
    rng = np.random.default_rng(seed)
    specs = [
        # mean, covariance
        (np.array([-3.0, -3.0]), np.array([[2.5, 0.8], [0.8, 0.4]])),  # long in x
        (np.array([3.0, -3.0]), np.array([[0.4, 0.8], [0.8, 2.5]])),  # long in y
        (np.array([0.0, 4.0]), np.array([[0.6, 0.0], [0.0, 0.6]])),  # isotropic
    ]
    Xs, ys = [], []
    for label, (mean, cov) in enumerate(specs):
        Xs.append(rng.multivariate_normal(mean, cov, size=n_per_class))
        ys.append(np.full(n_per_class, label))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # Stratified 80/20 split: shuffle within each class.
    train_idx, test_idx = [], []
    for label in range(3):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx.append(idx[:split])
        test_idx.append(idx[split:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main():
    X_train, y_train, X_test, y_test = make_data()
    print(f"Train size: {len(X_train)}, test size: {len(X_test)}")

    clf = QDAClassifier()
    clf.fit(X_train, y_train)

    train_acc = np.mean(clf.predict(X_train) == y_train)
    test_acc = np.mean(clf.predict(X_test) == y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    print("Per-class covariance determinants (fitted):")
    for label, cov in zip(clf.classes_, clf.covariances_):
        print(f"  class {label}: det = {np.linalg.det(cov):.4f}")

    preds = clf.predict(X_test)
    print("First 5 predictions vs true labels:")
    for pred, true in zip(preds[:5], y_test[:5]):
        print(f"  predicted={pred}, true={true}")

    assert test_acc >= 0.90, f"test accuracy {test_acc:.3f} below 0.90"


if __name__ == "__main__":
    main()
