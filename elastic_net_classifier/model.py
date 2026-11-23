"""Logistic regression with an elastic-net penalty, from scratch.

Trained with proximal gradient descent (ISTA): each step takes a gradient
step on the smooth part (logistic loss + L2 penalty) and then applies the
soft-thresholding proximal operator for the L1 penalty. Multiclass problems
are handled one-vs-rest. Features are standardised internally so the
penalty treats every feature on the same scale.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class ElasticNetClassifier:
    """Binary / multiclass logistic regression with L1 + L2 penalties.

    Parameters
    ----------
    alpha : float
        Overall penalty strength (must be >= 0; 0 gives plain logistic
        regression).
    l1_ratio : float
        Balance between the penalties: 1.0 is pure L1 (lasso), 0.0 is
        pure L2 (ridge). Must lie in [0, 1].
    max_iter : int
        Maximum proximal-gradient steps per one-vs-rest problem.
    tol : float
        Stop when the largest coefficient change in a step falls below this.
    learning_rate : float or None
        Step size for the gradient step; must be positive. ``None``
        (default) computes it as 1/L, where L is the Lipschitz constant
        of the smooth part (0.25 * largest eigenvalue of X'X/n + the
        L2 coefficient) — the largest step ISTA can safely take.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 max_iter: int = 1000, tol: float = 1e-4,
                 learning_rate: float | None = None):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol!r}")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}")
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None
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
        if self.coef_ is None:
            raise RuntimeError(
                "ElasticNetClassifier is not fitted yet; call fit() first"
            )

    # -- fitting --------------------------------------------------------
    def _fit_binary(self, X, y01):
        """Proximal gradient descent for one binary logistic problem.

        ``y01`` holds 0/1 labels. Returns (weights, intercept, n_iter).
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        lam1 = self.alpha * self.l1_ratio
        lam2 = self.alpha * (1.0 - self.l1_ratio)
        if self.learning_rate is None:
            # Largest safe ISTA step: 1 / Lipschitz constant of the
            # smooth part (logistic loss + L2 penalty).
            lam_max = float(np.linalg.eigvalsh(X.T @ X / n_samples)[-1])
            lr = 1.0 / (0.25 * lam_max + lam2)
        else:
            lr = self.learning_rate
        for it in range(1, self.max_iter + 1):
            p = _sigmoid(X @ w + b)
            grad_w = X.T @ (p - y01) / n_samples + lam2 * w
            grad_b = float(np.mean(p - y01))
            # Gradient step, then soft-thresholding for the L1 penalty.
            w_new = w - lr * grad_w
            w_new = np.sign(w_new) * np.maximum(np.abs(w_new) - lr * lam1, 0.0)
            b_new = b - lr * grad_b
            change = float(np.max(np.abs(w_new - w))) if n_features else 0.0
            w, b = w_new, b_new
            if change < self.tol:
                return w, b, it
        return w, b, self.max_iter

    def fit(self, X, y):
        """Fit the elastic-net logistic classifier."""
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)} labels"
            )
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError(
                f"need at least 2 classes to fit, got {len(self.classes_)}"
            )

        # Standardise so the penalty is scale-free; store for predict.
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Xs = (X - self.mean_) / self.scale_

        n_classes = len(self.classes_)
        self.coef_ = np.zeros((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)
        self.n_iter_ = np.zeros(n_classes, dtype=int)
        for k in range(n_classes):
            y01 = (y_idx == k).astype(float)
            w, b, it = self._fit_binary(Xs, y01)
            self.coef_[k] = w
            self.intercept_[k] = b
            self.n_iter_[k] = it
        return self

    # -- prediction -----------------------------------------------------
    def _standardise(self, X):
        return (X - self.mean_) / self.scale_

    def decision_function(self, X):
        """Log-odds per class, shape (n_samples, n_classes)."""
        self._check_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError(
                f"expected {self.coef_.shape[1]} features, got {X.shape[1]}"
            )
        Xs = self._standardise(X)
        return Xs @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        """Per-class probabilities (softmax over the one-vs-rest scores)."""
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predicted class labels."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
