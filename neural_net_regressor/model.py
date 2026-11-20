"""Multi-layer perceptron regressor, from scratch with numpy only.

One hidden layer with ReLU activation and a linear output, trained by
mini-batch gradient descent with backpropagation on mean squared error.
Optimizers: Adam (default, with bias correction) or plain SGD. Early
stopping on a held-out validation split restores the best weights.

Features and the target are standardized internally (mean / std stored
on the fitted model), so no manual scaling is needed before fitting.
"""

from __future__ import annotations

import numpy as np


class MLPRegressor:
    """MLP regressor with one hidden layer, ReLU, Adam/SGD, early stopping.

    Parameters
    ----------
    hidden_units : int, default 32
        Number of units in the single hidden layer.
    learning_rate : float, default 0.01
        Step size for the optimizer.
    epochs : int, default 500
        Maximum number of training epochs.
    batch_size : int, default 32
        Mini-batch size. Use a value >= n_samples for full-batch training.
    optimizer : {"adam", "sgd"}, default "adam"
        Optimizer to use.
    patience : int, default 20
        Epochs with no validation-loss improvement before early stopping.
        ``0`` disables early stopping (trains on all the data).
    random_state : int or None, default None
        Seed for weight initialization and batch shuffling. Same seed
        gives the same fitted model.
    l2 : float, default 0.0
        L2 (weight-decay) penalty applied to the weight matrices.
    validation_fraction : float, default 0.2
        Fraction of the data held out as validation for early stopping.
        Only used when ``patience > 0``.
    verbose : bool, default False
        If True, print the training/validation loss every 50 epochs.
    """

    def __init__(
        self,
        hidden_units: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 500,
        batch_size: int = 32,
        optimizer: str = "adam",
        patience: int = 20,
        random_state=None,
        l2: float = 0.0,
        validation_fraction: float = 0.2,
        verbose: bool = False,
    ):
        if isinstance(hidden_units, bool) or not isinstance(hidden_units, int):
            raise ValueError("hidden_units must be an integer")
        if hidden_units < 1:
            raise ValueError("hidden_units must be at least 1")
        if not isinstance(learning_rate, (int, float)) or isinstance(
            learning_rate, bool
        ):
            raise ValueError("learning_rate must be a number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if isinstance(epochs, bool) or not isinstance(epochs, int):
            raise ValueError("epochs must be an integer")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise ValueError("patience must be an integer")
        if patience < 0:
            raise ValueError("patience must be at least 0")
        if not isinstance(l2, (int, float)) or isinstance(l2, bool):
            raise ValueError("l2 must be a number")
        if l2 < 0:
            raise ValueError("l2 must be at least 0")
        if not isinstance(validation_fraction, (int, float)) or isinstance(
            validation_fraction, bool
        ):
            raise ValueError("validation_fraction must be a number")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("random_state must be an integer or None")

        self.hidden_units = hidden_units
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.patience = patience
        self.random_state = random_state
        self.l2 = float(l2)
        self.validation_fraction = float(validation_fraction)
        self.verbose = bool(verbose)

        self.history_: list[float] | None = None
        self.n_epochs_: int | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _init_weights(self, rng: np.random.Generator, n_features: int) -> None:
        # He initialization for the ReLU hidden layer, small init for output.
        self._W1 = rng.normal(
            0.0, np.sqrt(2.0 / n_features), size=(n_features, self.hidden_units)
        )
        self._b1 = np.zeros(self.hidden_units)
        self._W2 = rng.normal(
            0.0, np.sqrt(1.0 / self.hidden_units), size=(self.hidden_units, 1)
        )
        self._b2 = np.zeros(1)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z1 = X @ self._W1 + self._b1
        a1 = np.maximum(0.0, z1)
        out = a1 @ self._W2 + self._b2
        return out.ravel(), a1

    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        pred, _ = self._forward(X)
        return float(np.mean((pred - y) ** 2))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    @staticmethod
    def _validate_X_y(X, y) -> tuple[np.ndarray, np.ndarray]:
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("X must be a numeric array") from None
        try:
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("y must be a numeric array") from None
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples "
                f"(got {X.shape[0]} and {y.shape[0]})"
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values")
        return X, y

    def _validate_X(self, X) -> np.ndarray:
        self._check_fitted()
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("X must be a numeric array") from None
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")
        if X.shape[0] == 0:
            raise ValueError("cannot predict on empty data")
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features, got {X.shape[1]}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        return X

    # ------------------------------------------------------------------
    # fit / predict / score
    # ------------------------------------------------------------------
    def fit(self, X, y) -> "MLPRegressor":
        """Fit the MLP regressor. Returns self."""
        X, y = self._validate_X_y(X, y)
        n_samples, n_features = X.shape
        self._n_features = n_features
        rng = np.random.default_rng(self.random_state)

        # Standardize features and target (guard against zero std).
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0.0] = 1.0
        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std == 0.0:
            self._y_std = 1.0
        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std

        # Validation split for early stopping: last `validation_fraction`.
        use_early_stopping = self.patience > 0 and self.validation_fraction > 0.0
        if use_early_stopping:
            n_val = int(n_samples * self.validation_fraction)
            if n_val < 1 or n_samples - n_val < 2:
                use_early_stopping = False
        if use_early_stopping:
            X_train, y_train = Xs[:-n_val], ys[:-n_val]
            X_val, y_val = Xs[-n_val:], ys[-n_val:]
        else:
            X_train, y_train = Xs, ys
            X_val, y_val = None, None

        self._init_weights(rng, n_features)
        params = ["_W1", "_b1", "_W2", "_b2"]
        # Adam state.
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        min_delta = 1e-4  # smallest val-loss drop that counts as improvement

        best_val_loss = np.inf
        best_params = None
        epochs_no_improve = 0
        self.history_ = []

        n_train = X_train.shape[0]
        for epoch in range(1, self.epochs + 1):
            perm = rng.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                Xb, yb = X_train[idx], y_train[idx]
                m_batch = len(idx)

                z1 = Xb @ self._W1 + self._b1
                a1 = np.maximum(0.0, z1)
                out = (a1 @ self._W2 + self._b2).ravel()
                err = out - yb  # (m_batch,)

                # Gradients of MSE (linear output, ReLU hidden).
                d_out = (2.0 / m_batch) * err  # (m_batch,)
                grad_W2 = a1.T @ d_out.reshape(-1, 1)
                grad_b2 = np.array([d_out.sum()])
                d_a1 = np.outer(d_out, self._W2.ravel())
                d_z1 = d_a1 * (z1 > 0)
                grad_W1 = Xb.T @ d_z1
                grad_b1 = d_z1.sum(axis=0)
                grads = {
                    "_W1": grad_W1,
                    "_b1": grad_b1,
                    "_W2": grad_W2,
                    "_b2": grad_b2,
                }
                if self.l2 > 0.0:
                    grads["_W1"] += self.l2 * self._W1
                    grads["_W2"] += self.l2 * self._W2

                step += 1
                for p in params:
                    g = grads[p]
                    if self.optimizer == "adam":
                        m[p] = beta1 * m[p] + (1 - beta1) * g
                        v[p] = beta2 * v[p] + (1 - beta2) * (g * g)
                        m_hat = m[p] / (1 - beta1**step)
                        v_hat = v[p] / (1 - beta2**step)
                        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                    else:  # sgd
                        update = self.learning_rate * g
                    setattr(self, p, getattr(self, p) - update)

            train_loss = self._mse(X_train, y_train)
            self.history_.append(train_loss)
            if self.verbose and (epoch == 1 or epoch % 50 == 0):
                msg = f"epoch {epoch}: train MSE={train_loss:.6f}"
                if X_val is not None:
                    msg += f", val MSE={self._mse(X_val, y_val):.6f}"
                print(msg)

            if use_early_stopping:
                val_loss = self._mse(X_val, y_val)
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_params = {p: getattr(self, p).copy() for p in params}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break

        self.n_epochs_ = len(self.history_)
        if best_params is not None:
            for p in params:
                setattr(self, p, best_params[p])
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values. Returns a 1D array."""
        X = self._validate_X(X)
        Xs = (X - self._x_mean) / self._x_std
        pred, _ = self._forward(Xs)
        return pred * self._y_std + self._y_mean

    def score(self, X, y) -> float:
        """Return the R^2 (coefficient of determination) score."""
        self._check_fitted()
        try:
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("y must be a numeric array") from None
        X = self._validate_X(X)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples "
                f"(got {X.shape[0]} and {y.shape[0]})"
            )
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values")
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)
