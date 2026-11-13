"""Generates linearly separable synthetic data, fits logistic regression."""
import numpy as np

from logistic_regression.model import LogisticRegression


def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 100
    class0 = rng.normal(loc=[-2.0, -2.0], scale=1.0, size=(n, 2))
    class1 = rng.normal(loc=[2.0, 2.0], scale=1.0, size=(n, 2))
    x = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def main():
    x, y = make_data()
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    model = LogisticRegression(learning_rate=0.1, n_epochs=1000, l2=0.01).fit(
        x_train, y_train
    )
    preds = model.predict(x_test)
    accuracy = float(np.mean(preds == y_test))

    print(f"Train size: {len(y_train)}, test size: {len(y_test)}")
    print(f"Epochs run: {model.n_epochs_run_}, final loss: {model.loss_history_[-1]:.4f}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Weights: {np.round(model.coef_, 3).tolist()}, intercept: {model.intercept_:.3f}")
    print(f"First 5 predictions: {preds[:5].tolist()}")
    print(f"First 5 true labels: {y_test[:5].tolist()}")


if __name__ == "__main__":
    main()
