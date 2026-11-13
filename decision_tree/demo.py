"""Generates two synthetic 2D clusters, fits a decision tree, reports accuracy."""
import numpy as np

from decision_tree.model import DecisionTreeClassifier


def make_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 60
    class0 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n, 2))
    class1 = rng.normal(loc=[4.0, 3.0], scale=1.0, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_blobs()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = DecisionTreeClassifier(max_depth=4).fit(x_train, y_train)
    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}, max_depth=4")
    print(f"Tree depth: {model.depth}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
