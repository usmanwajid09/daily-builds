"""Partial least squares discriminant analysis (PLS-DA), from scratch.

PLS regression is fit with the NIPALS algorithm against one-hot class
targets, and a new sample is classified by the largest predicted target
value. PLS finds latent directions that maximise covariance between X
and the class targets, which makes it effective when features are
collinear — a case where ordinary classifiers can be unstable.
"""
from __future__ import annotations

import numpy as np


class PLSClassifier:
    """PLS-DA classifier using the NIPALS algorithm.

    Parameters
    ----------
    n_components : int
        Number of latent components. Must be at least 1 and at most
        ``min(n_samples - 1, n_features)``.
    max_iter : int
        Maximum NIPALS iterations per component.
    tol : float
        Convergence tolerance for the NIPALS weight vector.
    scale : bool
        Whether to standardise features to unit variance before fitting
        (recommended when features live on different scales).
    """

    def __init__(self, n_components: int = 2, max_iter: int = 500,
                 tol: float = 1e-6, scale: bool = True):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol!r}")
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scale = bool(scale)
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.x_weights_ = None
        self.x_loadings_ = None
        self.explained_variance_ratio_ = None

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
            raise RuntimeError("PLSClassifier is not fitted yet; call fit() first")

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        """Fit the PLS-DA classifier."""
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
        n_samples, n_features = X.shape
        max_comp = min(n_samples - 1, n_features)
        if self.n_components > max_comp:
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum "
                f"{max_comp} for {n_samples} samples and {n_features} features"
            )

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) if self.scale else np.ones(n_features)
        self.scale_[self.scale_ == 0] = 1.0
        Xc = (X - self.mean_) / self.scale_

        # One-hot targets, centred (PLS works on centred Y).
        Y = np.zeros((n_samples, len(self.classes_)))
        Y[np.arange(n_samples), y_idx] = 1.0
        self.y_mean_ = Y.mean(axis=0)
        Yc = Y - self.y_mean_

        # NIPALS: extract one latent component at a time.
        X_res, Y_res = Xc.copy(), Yc.copy()
        W, P, C = [], [], []
        x_var_total = float(np.sum(Xc ** 2))
        x_var_explained = []
        for _ in range(self.n_components):
            # Start from the column of Y with the largest variance.
            u = Y_res[:, int(np.argmax(np.var(Y_res, axis=0)))]
            w = np.zeros(n_features)
            for _ in range(self.max_iter):
                w_new = X_res.T @ u
                norm = np.linalg.norm(w_new)
                if norm == 0:
                    raise RuntimeError("NIPALS failed: zero weight vector")
                w_new /= norm
                t = X_res @ w_new
                denom = float(t @ t)
                if denom == 0:
                    raise RuntimeError("NIPALS failed: zero score vector")
                c = Y_res.T @ t / denom
                u_new = Y_res @ c / float(c @ c) if float(c @ c) > 0 else u
                if np.linalg.norm(w_new - w) < self.tol:
                    w, u = w_new, u_new
                    break
                w, u = w_new, u_new
            else:
                raise RuntimeError(
                    f"NIPALS did not converge within {self.max_iter} iterations"
                )
            t = X_res @ w
            denom = float(t @ t)
            p = X_res.T @ t / denom
            c = Y_res.T @ t / denom
            W.append(w)
            P.append(p)
            C.append(c)
            x_var_explained.append(float(t @ t) * float(p @ p) / x_var_total
                                   if x_var_total > 0 else 0.0)
            X_res = X_res - np.outer(t, p)
            Y_res = Y_res - np.outer(t, c)

        W = np.array(W).T          # (n_features, n_components)
        P = np.array(P).T          # (n_features, n_components)
        C = np.array(C).T          # (n_classes, n_components)
        self.x_weights_ = W
        self.x_loadings_ = P
        self.explained_variance_ratio_ = np.array(x_var_explained)
        # Regression coefficients in the original (standardised) space:
        # B = W (P'W)^-1 C'.
        try:
            B = W @ np.linalg.solve(P.T @ W, C.T)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"could not compute PLS coefficients: {exc}") from exc
        # Undo standardisation: coef on raw features, intercept from means.
        self.coef_ = (B / self.scale_[:, None]).T   # (n_classes, n_features)
        self.intercept_ = self.y_mean_ - self.mean_ @ self.coef_.T
        return self

    # -- prediction -----------------------------------------------------
    def decision_function(self, X):
        """Predicted one-hot target values, shape (n_samples, n_classes)."""
        self._check_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError(
                f"expected {self.coef_.shape[1]} features, got {X.shape[1]}"
            )
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """Predicted class labels."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
