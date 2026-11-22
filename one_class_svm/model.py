"""One-class SVM anomaly detection, from scratch with numpy only.

Learns a boundary around the training data using only the "normal"
class. With an RBF kernel, the score function is
``f(x) = sum_i alpha_i K(x_i, x)`` and points with ``f(x) < rho`` are
flagged as anomalies. The parameters ``alpha`` and ``rho`` are fit by
mini-batch stochastic gradient descent on the one-class primal
objective

    L = 0.5 * alpha^T K alpha
        + (1 / (nu * n)) * sum_i max(0, rho - f(x_i)) - rho,

with ``alpha`` projected back onto the box ``[0, 1 / (nu * n)]`` after
each step (the dual variables of a one-class SVM are non-negative and
bounded above by ``1 / (nu * n)``). After
optimisation, ``rho`` is set to the ``nu``-quantile of the training
scores, so roughly a fraction ``nu`` of the training points fall
outside the boundary.
"""
from __future__ import annotations

import numpy as np


def _rbf_kernel(a, b, gamma):
    # Squared Euclidean distances, computed stably.
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True).T
    dist_sq = np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)
    return np.exp(-gamma * dist_sq)


def _resolve_gamma(gamma, x):
    if isinstance(gamma, (int, float, np.integer, np.floating)):
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma!r}.")
        return float(gamma)
    if gamma == "scale":
        var = float(np.var(x))
        if var == 0:
            raise ValueError("gamma='scale' needs non-constant data (variance is 0).")
        return 1.0 / (x.shape[1] * var)
    if gamma == "auto":
        return 1.0 / x.shape[1]
    raise ValueError(f"gamma must be a positive float, 'scale' or 'auto', got {gamma!r}.")


def _validate_x(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {x.shape}.")
    if x.shape[0] == 0:
        raise ValueError("X must not be empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("X must contain only finite values.")
    return x


class OneClassSVM:
    """One-class SVM for anomaly detection (RBF kernel, SGD fit).

    Parameters
    ----------
    nu : float, default 0.1
        Upper bound on the fraction of training errors and lower bound
        on the fraction of support vectors. Must be in (0, 1].
    gamma : float, 'scale' or 'auto', default 'scale'
        RBF kernel coefficient. ``'scale'`` uses ``1 / (n_features *
        Var(X))`` and ``'auto'`` uses ``1 / n_features``.
    learning_rate : float, default 0.5
        Initial SGD step size. Must be > 0.
    epochs : int, default 300
        Number of passes over the training data. Must be >= 1.
    batch_size : int, default 32
        Mini-batch size for the hinge term. Must be >= 1.
    random_state : int or None, default None
        Seed for reproducibility.
    """

    def __init__(self, *, nu=0.1, gamma="scale", learning_rate=0.5,
                 epochs=300, batch_size=32, random_state=None):
        if not isinstance(nu, (int, float)) or not 0 < nu <= 1.0:
            raise ValueError(f"nu must be in (0, 1], got {nu!r}.")
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            raise ValueError(f"gamma must be a positive float, 'scale' or 'auto', got {gamma!r}.")
        if isinstance(gamma, (int, float, np.integer, np.floating)) and gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma!r}.")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}.")
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError(f"epochs must be an integer >= 1, got {epochs!r}.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {batch_size!r}.")
        self.nu = float(nu)
        self.gamma = gamma
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self.dual_coef_ = None
        self.rho_ = None
        self.gamma_ = None
        self.X_fit_ = None
        self.n_features_in_ = None
        self.support_ = None

    def fit(self, x):
        """Fit the one-class boundary on normal data. Returns self."""
        x = _validate_x(x)
        n, p = x.shape
        self.n_features_in_ = p
        self.gamma_ = _resolve_gamma(self.gamma, x)
        k = _rbf_kernel(x, x, self.gamma_)
        rng = np.random.default_rng(self.random_state)
        alpha = np.zeros(n)
        rho = 0.0
        c = 1.0 / (self.nu * n)
        # The quadratic term's curvature is the largest eigenvalue of K,
        # so scale the step size by it to keep SGD stable.
        lam_max = float(np.linalg.eigvalsh(k)[-1])
        base_lr = self.learning_rate / max(lam_max, 1e-12)
        step = 0
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = perm[start:start + self.batch_size]
                b = len(batch)
                f = k @ alpha
                viol = batch[f[batch] < rho]
                # Full gradient of the quadratic term; stochastic
                # gradient of the hinge term, scaled back up by n / b.
                grad_alpha = f.copy()
                if len(viol) > 0:
                    grad_alpha -= c * (n / b) * k[viol].sum(axis=0)
                grad_rho = c * (n / b) * len(viol) - 1.0
                step += 1
                lr = base_lr / (1.0 + 0.005 * step)
                alpha = np.clip(alpha - lr * grad_alpha, 0.0, c)
                rho = rho - lr * grad_rho
        self.dual_coef_ = alpha
        self.X_fit_ = x
        f_train = k @ alpha
        self.rho_ = float(np.quantile(f_train, self.nu))
        # SGD leaves tiny positive weights on most points; count as
        # support vectors only those with a non-negligible weight.
        self.support_ = np.flatnonzero(alpha > 1e-3)
        return self

    def _score_function(self, x):
        if self.dual_coef_ is None:
            raise ValueError("OneClassSVM is not fitted yet. Call fit() first.")
        x = _validate_x(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but the model was fit with {self.n_features_in_}.")
        k = _rbf_kernel(x, self.X_fit_, self.gamma_)
        return k @ self.dual_coef_

    def decision_function(self, x):
        """Signed distance to the boundary: > 0 means inlier."""
        return self._score_function(x) - self.rho_

    def score_samples(self, x):
        """Anomaly score: higher means more normal (same as decision_function)."""
        return self.decision_function(x)

    def predict(self, x):
        """Return +1 for inliers and -1 for outliers."""
        return np.where(self.decision_function(x) >= 0, 1, -1)
