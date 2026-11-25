"""Theil-Sen robust regression.

Median of pairwise slopes: resistant to gross outliers (breakdown ~29%).
Single-feature: exact median of all pairwise slopes.
Multivariate: per-feature Theil-Sen slopes fitted independently, then
refined by robust backfitting (each feature re-fit on its partial
residual, iterated to convergence), then the intercept is the median
residual. This is an approximation for d > 1 -- documented here and in
the README.
"""

import numpy as np


class TheilSenRegressor:
    """Theil-Sen robust linear regressor.

    Parameters
    ----------
    fit_intercept : bool, default True
        Whether to estimate the intercept. If False, intercept is fixed at 0.
    max_subpopulation : int or None, default None
        If set and the number of samples exceeds it, pairwise slopes are
        computed on a deterministic subsample of this many points
        (seeded by ``random_state``).
    random_state : int or None, default None
        Seed for the deterministic subsampling when ``max_subpopulation``
        is used.
    """

    def __init__(self, fit_intercept=True, max_subpopulation=None, random_state=None):
        self.fit_intercept = fit_intercept
        self.max_subpopulation = max_subpopulation
        self.random_state = random_state

    # -- validation -----------------------------------------------------
    @staticmethod
    def _validate(X, y):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise TypeError("X must be numeric")
        try:
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            raise TypeError("y must be numeric")
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must be non-empty")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or inf")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or inf")
        return X, y

    @staticmethod
    def _theil_sen_slope(x, y):
        """Median of all pairwise slopes for one feature."""
        n = len(x)
        slopes = []
        for i in range(n):
            dx = x[i + 1 :] - x[i]
            dy = y[i + 1 :] - y[i]
            mask = dx != 0
            if np.any(mask):
                slopes.append(dy[mask] / dx[mask])
        if not slopes:
            raise ValueError("All x values are identical: no valid slopes")
        return float(np.median(np.concatenate(slopes)))

    # -- API ------------------------------------------------------------
    def fit(self, X, y):
        X, y = self._validate(X, y)
        n, d = X.shape

        X_fit, y_fit = X, y
        if self.max_subpopulation is not None and n > self.max_subpopulation:
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(n, size=self.max_subpopulation, replace=False))
            X_fit, y_fit = X[idx], y[idx]

        coef = np.empty(d, dtype=float)
        for j in range(d):
            coef[j] = self._theil_sen_slope(X_fit[:, j], y_fit)

        if d > 1:
            # Backfitting: re-fit each feature's slope on its partial
            # residual so the other features' contributions are accounted
            # for (the independent per-feature slopes above are biased
            # when features are correlated / other features add variance).
            for _ in range(10):
                old = coef.copy()
                for j in range(d):
                    partial = y_fit - X_fit @ coef + coef[j] * X_fit[:, j]
                    coef[j] = self._theil_sen_slope(X_fit[:, j], partial)
                if np.max(np.abs(coef - old)) < 1e-8:
                    break
        self.coef_ = coef

        if self.fit_intercept:
            self.intercept_ = float(np.median(y - X @ coef))
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("TheilSenRegressor is not fitted yet; call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got {X.ndim}D")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.coef_.shape[0]}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or inf")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Coefficient of determination R^2."""
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        y_pred = self.predict(X)
        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
