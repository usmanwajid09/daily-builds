"""Support Vector Regression from scratch (NumPy only).

The primal SVR objective is

    0.5 * ||w||^2 + C * sum_i max(0, |y_i - (w.x_i + b)| - epsilon)

Errors smaller than ``epsilon`` cost nothing (the "epsilon-insensitive
tube"); only points outside the tube pull on the weights. The model is
trained with stochastic gradient descent: one sample at a time, shuffled
each epoch, using the subgradient of the loss plus the L2 penalty.
"""

import numpy as np


class SVRRegressor:
    """Linear Support Vector Regression trained by SGD on the primal.

    Parameters
    ----------
    epsilon : float, default=0.1
        Half-width of the insensitive tube; errors within it are ignored.
    C : float, default=1.0
        Penalty for errors outside the tube (larger = fit the data harder).
    learning_rate : float, default=0.01
        SGD step size.
    n_epochs : int, default=500
        Number of passes over the training data.
    random_state : int, default=42
        Seed for the per-epoch shuffling, so runs are reproducible.
    """

    def __init__(self, epsilon=0.1, C=1.0, learning_rate=0.01,
                 n_epochs=500, random_state=42):
        self.epsilon = epsilon
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    # -- validation -----------------------------------------------------
    def _validate_params(self):
        if self.epsilon is None or not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be a finite number >= 0, got {self.epsilon!r}.")
        if self.C is None or not np.isfinite(self.C) or self.C <= 0:
            raise ValueError(f"C must be a finite number > 0, got {self.C!r}.")
        if (self.learning_rate is None or not np.isfinite(self.learning_rate)
                or self.learning_rate <= 0):
            raise ValueError(
                f"learning_rate must be a finite number > 0, got {self.learning_rate!r}.")
        if self.n_epochs is None or int(self.n_epochs) != self.n_epochs or self.n_epochs < 1:
            raise ValueError(f"n_epochs must be an integer >= 1, got {self.n_epochs!r}.")

    @staticmethod
    def _validate_data(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must be non-empty.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: {X.shape[0]} vs {y.shape[0]}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values (no NaN or inf).")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values (no NaN or inf).")
        return X, y

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        """Fit the SVR model. Returns self."""
        self._validate_params()
        X, y = self._validate_data(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Standardise features (store for use in predict).
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # constant feature: leave as-is
        Xs = (X - self.mean_) / self.std_

        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        # Per-sample penalty so the total loss matches the batch objective.
        c_sample = self.C / n_samples

        for _ in range(self.n_epochs):
            order = rng.permutation(n_samples)
            for i in order:
                xi, yi = Xs[i], y[i]
                residual = yi - (xi @ self.coef_ + self.intercept_)
                # Per-sample gradient of the objective divided by n, so one
                # epoch of updates matches the batch gradient in expectation:
                # the L2 term and the loss term are scaled by the same 1/n.
                grad_w = self.coef_ / n_samples
                grad_b = 0.0
                if abs(residual) > self.epsilon:
                    sign = 1.0 if residual > 0 else -1.0
                    grad_w -= c_sample * sign * xi
                    grad_b -= c_sample * sign
                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b
        return self

    # -- prediction -----------------------------------------------------
    def predict(self, X):
        """Predict target values for X. Returns a 1D array."""
        if not hasattr(self, "coef_"):
            raise ValueError("SVRRegressor is not fitted yet; call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values (no NaN or inf).")
        Xs = (X - self.mean_) / self.std_
        return Xs @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Return the coefficient of determination R² of the prediction."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
