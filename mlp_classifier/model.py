"""Small multi-layer perceptron classifier, from scratch with numpy only.

A feed-forward network with one hidden layer: an affine transform,
ReLU activation, a second affine transform, and a softmax output
trained with cross-entropy loss by mini-batch stochastic gradient
descent. Weights use He initialisation (scaled for ReLU), inputs are
standardised before training, and the trained mean/std are stored so
predictions use the same scaling. Training shuffles the data each
epoch with a seeded RNG, so results are reproducible for a given
``random_state``. The network is deliberately small — it exists to
show the full forward/backward pass in readable numpy, not to compete
with a framework — but it genuinely learns non-linear boundaries such
as two moons.
"""
from __future__ import annotations

import numpy as np


class MLPClassifier:
    """Multi-layer perceptron classifier with one hidden layer.

    Parameters
    ----------
    hidden_size : int
        Number of units in the hidden layer. Must be at least 1.
    learning_rate : float
        Step size for gradient descent. Must be positive.
    epochs : int
        Number of full passes over the training data. Must be at least 1.
    batch_size : int
        Mini-batch size. Must be at least 1.
    l2 : float
        L2 weight-decay strength. Must be non-negative.
    random_state : int, optional
        Seed for weight initialisation and shuffling. Fixes results
        when given.
    """

    def __init__(self, hidden_size: int = 16, learning_rate: float = 0.05,
                 epochs: int = 200, batch_size: int = 32, l2: float = 0.0,
                 random_state: int | None = None):
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) \
                or hidden_size < 1:
            raise ValueError("hidden_size must be an integer of at least 1")
        if not isinstance(learning_rate, (int, float)) \
                or isinstance(learning_rate, bool) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError("epochs must be an integer of at least 1")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) \
                or batch_size < 1:
            raise ValueError("batch_size must be an integer of at least 1")
        if not isinstance(l2, (int, float)) or isinstance(l2, bool) or l2 < 0:
            raise ValueError("l2 must be a non-negative number")
        if random_state is not None and (
                not isinstance(random_state, int) or isinstance(random_state, bool)):
            raise ValueError("random_state must be an integer or None")
        self.hidden_size = hidden_size
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = float(l2)
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("MLP needs at least 2 classes")
        n, n_features = x.shape
        n_classes = classes.size
        y_idx = np.searchsorted(classes, y)

        rng = np.random.default_rng(self.random_state)
        self._x_mean = x.mean(axis=0)
        self._x_std = x.std(axis=0)
        self._x_std[self._x_std == 0.0] = 1.0
        xs = (x - self._x_mean) / self._x_std

        # He initialisation.
        self._w1 = rng.normal(scale=np.sqrt(2.0 / n_features),
                              size=(n_features, self.hidden_size))
        self._b1 = np.zeros(self.hidden_size)
        self._w2 = rng.normal(scale=np.sqrt(2.0 / self.hidden_size),
                              size=(self.hidden_size, n_classes))
        self._b2 = np.zeros(n_classes)
        self.classes_ = classes

        self.loss_history_ = []
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = xs[idx], y_idx[idx]
                loss = self._train_step(xb, yb)
                epoch_loss += loss * len(idx)
            self.loss_history_.append(epoch_loss / n)
        return self

    def _train_step(self, xb: np.ndarray, yb: np.ndarray) -> float:
        m = len(yb)
        # Forward pass.
        z1 = xb @ self._w1 + self._b1
        a1 = np.maximum(z1, 0.0)  # ReLU
        logits = a1 @ self._w2 + self._b2
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -float(np.mean(np.log(probs[np.arange(m), yb] + 1e-15)))

        # Backward pass (cross-entropy + softmax gradient).
        dlogits = probs.copy()
        dlogits[np.arange(m), yb] -= 1.0
        dlogits /= m
        dw2 = a1.T @ dlogits + self.l2 * self._w2
        db2 = dlogits.sum(axis=0)
        da1 = dlogits @ self._w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = xb.T @ dz1 + self.l2 * self._w1
        db1 = dz1.sum(axis=0)

        self._w2 -= self.learning_rate * dw2
        self._b2 -= self.learning_rate * db2
        self._w1 -= self.learning_rate * dw1
        self._b1 -= self.learning_rate * db1
        return loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities (rows sum to 1)."""
        if self.classes_ is None:
            raise ValueError("model is not fitted yet; call fit() first")
        x = self._check_predict_input(x)
        xs = (x - self._x_mean) / self._x_std
        a1 = np.maximum(xs @ self._w1 + self._b1, 0.0)
        logits = a1 @ self._w2 + self._b2
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("model is not fitted yet; call fit() first")
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(x) == np.asarray(y)))

    def _check_predict_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[1] != self._w1.shape[0]:
            raise ValueError(
                f"x has {x.shape[1]} features but the model was fitted on "
                f"{self._w1.shape[0]}"
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        return x
