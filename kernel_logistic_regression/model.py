"""Kernel logistic regression with an RBF kernel, from scratch.

Fits logistic regression in the dual: the decision function is a weighted
sum of RBF kernel evaluations against the training points,
``f(x) = sum_i alpha_i * k(x_i, x) + b``, with the coefficients trained by
gradient descent on the L2-regularised logistic loss. The kernel trick
lets it learn non-linear decision boundaries without ever building the
feature map explicitly.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class KernelLogisticRegression:
    """Binary kernel logistic regression with an RBF kernel.

    Parameters
    ----------
    gamma : float
        RBF kernel width: ``k(x, z) = exp(-gamma * ||x - z||^2)``.
        Must be positive; larger values make the kernel more local.
    C : float
        Inverse regularisation strength (must be positive). Larger C
        fits the training data harder; smaller C regularises more.
    max_iter : int
        Maximum gradient-descent steps.
    tol : float
        Stop when the largest coefficient change in a step falls below this.
    learning_rate : float or None
        Gradient step size. ``None`` (default) uses 1/L with L the
        Lipschitz constant of the loss gradient.
    """

    def __init__(self, gamma: float = 1.0, C: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-4,
                 learning_rate: float | None = None):
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma!r}")
        if not np.isfinite(C) or C <= 0:
            raise ValueError(f"C must be positive, got {C!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol!r}")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}")
        self.gamma = float(gamma)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.dual_coef_ = None
        self.intercept_ = None
        self.X_train_ = None
        self.classes_ = None
        self.n_iter_ = None

    # -- validation -----------------------------------------------------
    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have at least one sample and one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _check_fitted(self):
        if self.dual_coef_ is None:
            raise RuntimeError(
                "KernelLogisticRegression is not fitted yet; call fit() first"
            )

    def _kernel(self, A, B):
        """RBF kernel matrix between rows of A and rows of B."""
        sq_a = np.sum(A ** 2, axis=1, keepdims=True)
        sq_b = np.sum(B ** 2, axis=1)[None, :]
        sq_dist = np.maximum(sq_a + sq_b - 2.0 * (A @ B.T), 0.0)
        return np.exp(-self.gamma * sq_dist)

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        """Fit the kernel logistic regression (binary only)."""
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)} labels"
            )
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(
                "KernelLogisticRegression is binary only; "
                f"got {len(self.classes_)} classes"
            )

        n_samples = X.shape[0]
        K = self._kernel(X, X)
        y01 = y_idx.astype(float)
        lam = 1.0 / (self.C * n_samples)  # L2 penalty on dual coefficients
        alpha = np.zeros(n_samples)
        b = 0.0
        if self.learning_rate is None:
            # Lipschitz constant of the gradient: 0.25 * ||K||^2 / n + lam.
            lam_max = float(np.linalg.eigvalsh(K)[-1])
            lr = 1.0 / (0.25 * lam_max ** 2 / n_samples + lam)
        else:
            lr = self.learning_rate
        for it in range(1, self.max_iter + 1):
            p = _sigmoid(K @ alpha + b)
            grad_alpha = K.T @ (p - y01) / n_samples + lam * alpha
            grad_b = float(np.mean(p - y01))
            alpha_new = alpha - lr * grad_alpha
            b_new = b - lr * grad_b
            change = float(np.max(np.abs(alpha_new - alpha)))
            alpha, b = alpha_new, b_new
            if change < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter
        self.dual_coef_ = alpha
        self.intercept_ = float(b)
        self.X_train_ = X
        return self

    # -- prediction -----------------------------------------------------
    def decision_function(self, X):
        """Log-odds of the positive class, shape (n_samples,)."""
        self._check_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"expected {self.X_train_.shape[1]} features, got {X.shape[1]}"
            )
        return self._kernel(X, self.X_train_) @ self.dual_coef_ + self.intercept_

    def predict_proba(self, X):
        """Probabilities of [negative, positive] class, shape (n_samples, 2)."""
        scores = self.decision_function(X)
        p1 = _sigmoid(scores)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Predicted class labels."""
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
