"""Partial least squares (PLS) regression via NIPALS, from scratch.

PLS finds latent components that maximise the covariance between the
predictors ``X`` and the responses ``y``, which makes it stable when
the features are strongly collinear — exactly the situation where
ordinary least squares becomes unstable. This implementation is the
NIPALS algorithm for PLS2 (multi-output ``y`` is supported, as is 1D
``y``), using numpy only — no scikit-learn.

The data are always centred; with ``scale=True`` (the default) ``X``
and ``y`` are additionally standardised to unit variance. A column
with zero variance gets a scale factor of 1 instead of dividing by
zero, so constant columns are left alone rather than crashing the fit.
"""

from __future__ import annotations

import numpy as np


class PLSRegression:
    """Partial least squares regression via the NIPALS algorithm.

    Args:
        n_components: number of latent components to extract. Must be
            at least 1 and at most ``min(n_samples - 1, n_features)``.
        scale: if True (default), standardise ``X`` and ``y`` to unit
            variance after centring. Constant (zero-variance) columns
            are left unscaled (scale factor 1).
        max_iter: maximum NIPALS iterations per component.
        tol: convergence tolerance on the relative change of the
            X-scores ``t`` between NIPALS iterations.
    """

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
    ):
        if isinstance(n_components, bool) or not isinstance(
            n_components, (int, np.integer)
        ):
            raise ValueError(
                f"n_components must be an integer >= 1, got {n_components!r}"
            )
        if int(n_components) < 1:
            raise ValueError(
                f"n_components must be >= 1, got {n_components}"
            )
        if isinstance(max_iter, bool) or not isinstance(
            max_iter, (int, np.integer)
        ):
            raise ValueError(f"max_iter must be an integer >= 1, got {max_iter!r}")
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if isinstance(tol, bool) or not isinstance(tol, (int, float, np.floating)):
            raise ValueError(f"tol must be a positive number, got {tol!r}")
        if not np.isfinite(tol) or float(tol) <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")

        self.n_components = int(n_components)
        self.scale = bool(scale)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.x_weights_: np.ndarray | None = None
        self.x_loadings_: np.ndarray | None = None
        self.y_loadings_: np.ndarray | None = None
        self.x_scores_: np.ndarray | None = None
        self.y_scores_: np.ndarray | None = None
        self.x_rotations_: np.ndarray | None = None
        self.n_components_: int | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("PLSRegression is not fitted yet — call fit() first.")

    @staticmethod
    def _as_2d_float(arr, name: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must not contain NaN or inf values")
        return arr

    def _validate_X(self, X) -> np.ndarray:
        X = self._as_2d_float(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted "
                f"with {self.n_features_in_} features"
            )
        return X

    # ------------------------------------------------------------------
    # fit / predict / score / transform
    # ------------------------------------------------------------------
    def fit(self, X, y) -> "PLSRegression":
        """Fit the PLS model to training data.

        Args:
            X: 2D array of shape (n_samples, n_features).
            y: 1D array of shape (n_samples,) or 2D array of shape
                (n_samples, n_targets).

        Returns:
            self, with ``coef_``, ``intercept_`` and the NIPALS
            matrices (``x_weights_``, ``x_loadings_``, ``y_loadings_``,
            ``x_scores_``, ``y_scores_``, ``x_rotations_``) set.
        """
        X = self._as_2d_float(X, "X")
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim != 2:
            raise ValueError(f"y must be 1D or 2D, got shape {y_arr.shape}")
        if y_arr.size == 0:
            raise ValueError("y must be non-empty")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y must not contain NaN or inf values")
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: X has {X.shape[0]} rows "
                f"but y has {y_arr.shape[0]}"
            )

        n_samples, n_features = X.shape
        n_targets = y_arr.shape[1]
        max_components = min(n_samples - 1, n_features)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large: it must be "
                f"<= min(n_samples - 1, n_features) = {max_components} "
                f"for {n_samples} samples and {n_features} features"
            )

        self.n_features_in_ = n_features
        self._y_1d = y_arr.shape[1] == 1 and np.asarray(y).ndim == 1

        # Centre (and optionally scale) the data.
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y_arr.mean(axis=0)
        Xc = X - self.x_mean_
        Yc = y_arr - self.y_mean_
        if self.scale:
            x_std = Xc.std(axis=0)
            y_std = Yc.std(axis=0)
            # Constant columns keep a scale factor of 1 (no division by 0).
            x_std = np.where(x_std > 0, x_std, 1.0)
            y_std = np.where(y_std > 0, y_std, 1.0)
        else:
            x_std = np.ones(n_features)
            y_std = np.ones(n_targets)
        self.x_std_ = x_std
        self.y_std_ = y_std
        Xs = Xc / x_std
        Ys = Yc / y_std

        # NIPALS, one component at a time.
        W = np.zeros((n_features, self.n_components))
        P = np.zeros((n_features, self.n_components))
        Q = np.zeros((n_targets, self.n_components))
        T = np.zeros((n_samples, self.n_components))
        U = np.zeros((n_samples, self.n_components))
        n_extracted = 0
        # Scale of the (centred/scaled) X matrix: a component whose scores
        # are negligible next to this is numerical residue from an already
        # exhausted X, and extracting it would amplify noise via q.
        x_scale = float(np.linalg.norm(Xs))

        X_res, Y_res = Xs.copy(), Ys.copy()
        for a in range(self.n_components):
            # Initialise u from the Y column with the largest variance
            # (the first column if all are equal, e.g. single target).
            variances = np.var(Y_res, axis=0)
            u = Y_res[:, int(np.argmax(variances))].copy()
            if not np.any(u):
                # Nothing left to explain in Y — stop extracting.
                break
            t_old = np.zeros(n_samples)
            w = q = t = None
            for _ in range(self.max_iter):
                w = X_res.T @ u
                w_norm = np.linalg.norm(w)
                if w_norm == 0.0:
                    break
                w = w / w_norm
                t = X_res @ w
                t_norm_sq = float(t @ t)
                if t_norm_sq == 0.0:
                    break
                q = Y_res.T @ t / t_norm_sq
                q_norm_sq = float(q @ q)
                if q_norm_sq == 0.0:
                    break
                u = Y_res @ q / q_norm_sq
                if np.linalg.norm(t - t_old) <= self.tol * max(
                    np.linalg.norm(t), 1e-12
                ):
                    break
                t_old = t
            if w is None or t is None or q is None:
                break
            t_norm_sq = float(t @ t)
            if t_norm_sq == 0.0 or np.linalg.norm(t) <= 1e-10 * x_scale:
                # No real signal left in X (rank exhausted) — stop
                # rather than fitting numerical residue.
                break
            p = X_res.T @ t / t_norm_sq

            W[:, a] = w
            P[:, a] = p
            Q[:, a] = q
            T[:, a] = t
            U[:, a] = u
            n_extracted += 1

            # Deflate: remove this component from X and Y.
            X_res = X_res - np.outer(t, p)
            Y_res = Y_res - np.outer(t, q)

        self.n_components_ = n_extracted
        self.x_weights_ = W[:, :n_extracted]
        self.x_loadings_ = P[:, :n_extracted]
        self.y_loadings_ = Q[:, :n_extracted]
        self.x_scores_ = T[:, :n_extracted]
        self.y_scores_ = U[:, :n_extracted]

        # Rotations R = W (P^T W)^-1, then coefficients in scaled space
        # B_scaled = R Q^T, converted back to the original space.
        if n_extracted == 0:
            B_scaled = np.zeros((n_features, n_targets))
            self.x_rotations_ = np.zeros((n_features, 0))
        else:
            PtW = self.x_loadings_.T @ self.x_weights_
            self.x_rotations_ = self.x_weights_ @ np.linalg.pinv(PtW)
            B_scaled = self.x_rotations_ @ self.y_loadings_.T

        self.coef_ = B_scaled * (self.y_std_ / self.x_std_[:, None])
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X) -> np.ndarray:
        """Predict targets for X.

        Returns a 1D array when the model was fitted on 1D ``y``,
        otherwise a 2D array of shape (n_samples, n_targets).
        """
        self._check_fitted()
        X = self._validate_X(X)
        y_pred = X @ self.coef_ + self.intercept_
        if self._y_1d:
            return y_pred[:, 0]
        return y_pred

    def transform(self, X) -> np.ndarray:
        """Project X onto the latent components (X-scores).

        Returns an array of shape (n_samples, n_components).
        """
        self._check_fitted()
        X = self._validate_X(X)
        Xc = (X - self.x_mean_) / self.x_std_
        return Xc @ self.x_rotations_

    def score(self, X, y) -> float:
        """Return the coefficient of determination R^2.

        For multi-output ``y`` this is the mean R^2 over targets.
        A target with zero variance scores 1.0 if predicted exactly,
        else 0.0.
        """
        self._check_fitted()
        X = self._validate_X(X)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim != 2:
            raise ValueError(f"y must be 1D or 2D, got shape {y_arr.shape}")
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: X has {X.shape[0]} rows "
                f"but y has {y_arr.shape[0]}"
            )
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y must not contain NaN or inf values")

        y_pred = self.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        r2s = []
        for j in range(y_arr.shape[1]):
            ss_res = float(np.sum((y_arr[:, j] - y_pred[:, j]) ** 2))
            ss_tot = float(np.sum((y_arr[:, j] - y_arr[:, j].mean()) ** 2))
            if ss_tot == 0.0:
                r2s.append(1.0 if ss_res == 0.0 else 0.0)
            else:
                r2s.append(1.0 - ss_res / ss_tot)
        return float(np.mean(r2s))
