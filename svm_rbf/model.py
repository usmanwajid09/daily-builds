"""Kernel SVM classifier with an RBF (Gaussian) kernel, from scratch.

The dual problem is solved with Platt's SMO (sequential minimal
optimisation): at each step a pair of Lagrange multipliers is updated
analytically under the box constraints 0 <= alpha <= C and the linear
equality constraint sum(alpha * y) == 0. Only numpy is used.
"""
from __future__ import annotations

import numpy as np


class RBFKernelSVM:
    """Binary SVM with a radial-basis-function kernel.

    Parameters
    ----------
    C : float
        Regularisation strength; must be positive. Larger C penalises
        margin violations more and fits the training data harder.
    gamma : {"scale", "auto"} or float
        Kernel coefficient. ``"scale"`` means ``1 / (n_features * X.var())``
        and ``"auto"`` means ``1 / n_features``. A positive float is used
        as-is.
    max_iter : int
        Maximum number of full passes over the training set during SMO.
    tol : float
        KKT tolerance: a multiplier is only updated when its KKT violation
        exceeds this value.
    random_state : int or None
        Seed for the random choice of the second multiplier in a pair.
        Fixes the seed for reproducibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int | None = None,
    ):
        if not isinstance(C, (int, float)) or isinstance(C, bool) or C <= 0:
            raise ValueError("C must be a positive number, got %r" % (C,))
        if isinstance(gamma, str):
            if gamma not in ("scale", "auto"):
                raise ValueError(
                    "gamma must be 'scale', 'auto', or a positive float, "
                    "got %r" % (gamma,)
                )
        elif isinstance(gamma, (int, float)) and not isinstance(gamma, bool):
            if gamma <= 0:
                raise ValueError("gamma must be positive, got %r" % (gamma,))
        else:
            raise ValueError(
                "gamma must be 'scale', 'auto', or a positive float, "
                "got %r" % (gamma,)
            )
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1, got %r" % (max_iter,))
        if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol <= 0:
            raise ValueError("tol must be a positive number, got %r" % (tol,))
        self.C = float(C)
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = float(tol)
        self.random_state = random_state
        self._fitted = False

    # -- kernel ---------------------------------------------------------
    def _rbf_kernel(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Gaussian kernel matrix between rows of x and rows of z."""
        x_sq = np.sum(x ** 2, axis=1)[:, None]
        z_sq = np.sum(z ** 2, axis=1)[None, :]
        sq_dist = np.maximum(x_sq + z_sq - 2.0 * (x @ z.T), 0.0)
        return np.exp(-self._gamma_value * sq_dist)

    # -- fitting --------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFKernelSVM":
        """Fit the SVM on training data X with binary labels y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples "
                "(got %d and %d)" % (X.shape[0], y.shape[0])
            )
        if X.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values (no NaN or inf)")
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                "y must contain exactly 2 distinct classes, got %d"
                % classes.shape[0]
            )

        n_samples, n_features = X.shape
        self.classes_ = classes
        # Map the two classes to -1 / +1 internally.
        y_signed = np.where(y == classes[0], -1.0, 1.0)

        # Resolve gamma.
        if self.gamma == "scale":
            var = X.var()
            self._gamma_value = (
                1.0 / (n_features * var) if var > 0 else 1.0 / n_features
            )
        elif self.gamma == "auto":
            self._gamma_value = 1.0 / n_features
        else:
            self._gamma_value = float(self.gamma)

        K = self._rbf_kernel(X, X)
        alpha, b = self._smo(K, y_signed)

        self.dual_coef_ = alpha
        self.intercept_ = float(b)
        sv_mask = alpha > 1e-8
        self.support_vectors_ = X[sv_mask]
        self.support_vector_labels_ = y_signed[sv_mask]
        self._support_mask = sv_mask
        self._X_train = X
        self._y_signed = y_signed
        self.n_support_ = int(sv_mask.sum())
        self.n_features_in_ = n_features
        self._fitted = True
        return self

    def _smo(self, K: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Platt's SMO on the dual, with an incrementally updated f cache.

        f[i] is the current decision value (without the +1/-1 label applied)
        for training point i, i.e. sum_j alpha_j y_j K_ji + b.
        """
        n = y.shape[0]
        rng = np.random.default_rng(self.random_state)
        alpha = np.zeros(n)
        b = 0.0
        f = np.zeros(n)  # f = K @ (alpha * y) + b, updated incrementally
        C = self.C
        tol = self.tol

        num_changed = 0
        examine_all = True
        passes = 0
        b_holder = [0.0]  # single holder so _take_step sees the running bias
        while passes < self.max_iter and (num_changed > 0 or examine_all):
            num_changed = 0
            if examine_all:
                indices = range(n)
            else:
                indices = [i for i in range(n) if 0.0 < alpha[i] < C]
            for i in indices:
                E_i = f[i] - y[i]
                r_i = y[i] * E_i
                if (r_i < -tol and alpha[i] < C) or (r_i > tol and alpha[i] > 0):
                    # Second multiplier: maximise |E_i - E_j|; fall back to
                    # a random j if that pair cannot be updated.
                    E = f - y
                    j = int(np.argmax(np.abs(E - E_i)))
                    if j == i:
                        j = int(rng.integers(n))
                        if j == i:
                            j = (i + 1) % n
                    if self._take_step(i, j, K, y, alpha, f, b_holder):
                        num_changed += 1
                    else:
                        # Try a few random partners before giving up on i.
                        for _ in range(5):
                            j2 = int(rng.integers(n))
                            if j2 == i:
                                continue
                            if self._take_step(i, j2, K, y, alpha, f, b_holder):
                                num_changed += 1
                                break
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            passes += 1

        # Recompute the bias from the margin support vectors: for a margin
        # SV, y_i * (sum_j alpha_j y_j K_ji + b) == 1, so average the implied
        # b over all of them (fall back to all SVs if none are on the margin).
        alpha = np.clip(alpha, 0.0, C)  # remove float artefacts like -1e-18
        margin = (alpha > 1e-8) & (alpha < C - 1e-8)
        use = margin if margin.any() else (alpha > 1e-8)
        if use.any():
            b = float(
                np.mean(y[use] - K[use] @ (alpha * y))
            )
        return alpha, b

    def _take_step(
        self,
        i: int,
        j: int,
        K: np.ndarray,
        y: np.ndarray,
        alpha: np.ndarray,
        f: np.ndarray,
        b_holder: list,
    ) -> bool:
        """One analytic SMO update of the pair (i, j). Mutates alpha and f.

        ``b_holder`` is a one-element list carrying the bias, so the caller
        sees the updated value. Returns True if the pair changed.
        """
        if i == j:
            return False
        b = b_holder[0]
        C = self.C
        E_i = f[i] - y[i]
        E_j = f[j] - y[j]
        alpha_i_old, alpha_j_old = alpha[i], alpha[j]

        if y[i] != y[j]:
            L = max(0.0, alpha_j_old - alpha_i_old)
            H = min(C, C + alpha_j_old - alpha_i_old)
        else:
            L = max(0.0, alpha_i_old + alpha_j_old - C)
            H = min(C, alpha_i_old + alpha_j_old)
        if L >= H:
            return False

        eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
        if eta <= 1e-12:
            return False
        alpha_j_new = alpha_j_old + y[j] * (E_i - E_j) / eta
        alpha_j_new = min(max(alpha_j_new, L), H)
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return False
        alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

        # Bias from the two updated multipliers.
        b1 = (
            b
            - E_i
            - y[i] * (alpha_i_new - alpha_i_old) * K[i, i]
            - y[j] * (alpha_j_new - alpha_j_old) * K[i, j]
        )
        b2 = (
            b
            - E_j
            - y[i] * (alpha_i_new - alpha_i_old) * K[i, j]
            - y[j] * (alpha_j_new - alpha_j_old) * K[j, j]
        )
        if 0.0 < alpha_i_new < C:
            b_new = b1
        elif 0.0 < alpha_j_new < C:
            b_new = b2
        else:
            b_new = 0.5 * (b1 + b2)

        # Update the cached decision values: f += d_i*y_i*K[:,i] + d_j*y_j*K[:,j]
        # plus the bias shift.
        f += (alpha_i_new - alpha_i_old) * y[i] * K[:, i]
        f += (alpha_j_new - alpha_j_old) * y[j] * K[:, j]
        f += b_new - b
        alpha[i], alpha[j] = alpha_i_new, alpha_j_new
        b_holder[0] = b_new
        return True

    # -- prediction -----------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has %d features, expected %d"
                % (X.shape[1], self.n_features_in_)
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values (no NaN or inf)")
        return X

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the separating hyperplane (1D array)."""
        X = self._check_X(X)
        sv = self.support_vectors_
        K = self._rbf_kernel(X, sv)
        return K @ (self.dual_coef_[self._support_mask] * self.support_vector_labels_) + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels for X."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
