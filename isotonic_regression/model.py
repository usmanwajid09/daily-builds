"""Isotonic regression via the pool adjacent violators algorithm (PAVA).

Isotonic regression fits a non-decreasing (or non-increasing) stepwise
function to data: it finds the monotone function closest to the
observed ``y`` in least-squares terms. It is the standard tool for
calibrating classifier probabilities and for fitting any relationship
known to be monotone (dose-response, price vs. demand, ...).

The pool adjacent violators algorithm (PAVA) works on the data sorted
by ``x``: it walks along the points, and whenever a point violates
monotonicity with its predecessor, the two are pooled into a block and
replaced by their (weighted) average, repeating until the block
averages are monotone. Predictions are made by linear interpolation
between the fitted points.
"""

from __future__ import annotations

import numpy as np


def _pava(y, weights, increasing: bool):
    """Pool adjacent violators algorithm. Returns block-averaged y."""
    n = len(y)
    values = y.astype(float).copy()
    w = weights.astype(float).copy()
    # Stack-based PAVA: maintain blocks in order, merging violators.
    block_start = []
    block_value = []
    block_weight = []
    for j in range(n):
        val, wt = values[j], w[j]
        start = j
        while block_value and (
            (block_value[-1] > val) if increasing else (block_value[-1] < val)
        ):
            prev_val = block_value.pop()
            prev_wt = block_weight.pop()
            prev_start = block_start.pop()
            val = (val * wt + prev_val * prev_wt) / (wt + prev_wt)
            wt = wt + prev_wt
            start = prev_start
        block_start.append(start)
        block_value.append(val)
        block_weight.append(wt)
    # Expand blocks back to per-point fitted values.
    fitted = np.empty(n)
    for k, start in enumerate(block_start):
        end = block_start[k + 1] if k + 1 < len(block_start) else n
        fitted[start:end] = block_value[k]
    return fitted


class IsotonicRegression:
    """Isotonic regression (monotone least-squares fit).

    Parameters
    ----------
    increasing : bool
        If True (default), fit a non-decreasing function; otherwise a
        non-increasing one.
    out_of_bounds : {"nan", "clip", "raise"}
        How to handle predictions outside the training ``x`` range:
        return NaN, clip to the nearest fitted value, or raise
        ``ValueError``. Default "nan".
    """

    def __init__(self, increasing: bool = True, out_of_bounds: str = "nan"):
        if out_of_bounds not in ("nan", "clip", "raise"):
            raise ValueError(
                f"out_of_bounds must be 'nan', 'clip' or 'raise', "
                f"got {out_of_bounds!r}"
            )
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds
        self.X_ = None
        self.y_ = None
        self.f_ = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 1:
            raise ValueError(f"X must be 1D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y must not be empty")
        if len(X) != len(y):
            raise ValueError(
                f"X and y have inconsistent lengths: {len(X)} vs {len(y)}"
            )
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must not contain NaN or infinite values")
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != y.shape:
                raise ValueError(
                    f"sample_weight must have shape {y.shape}, got {weights.shape}"
                )
            if np.any(weights < 0):
                raise ValueError("sample_weight must not be negative")
        else:
            weights = np.ones_like(y)
        order = np.argsort(X, kind="stable")
        X_sorted = X[order]
        y_sorted = y[order]
        w_sorted = weights[order]
        fitted = _pava(y_sorted, w_sorted, self.increasing)
        self.X_ = X_sorted
        self.y_ = fitted
        self.f_ = fitted
        return self

    def predict(self, X):
        if self.X_ is None:
            raise ValueError("IsotonicRegression is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 1:
            raise ValueError(f"X must be 1D, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values")
        below = X < self.X_[0]
        above = X > self.X_[-1]
        if self.out_of_bounds == "raise" and (below.any() or above.any()):
            raise ValueError("X contains values outside the training range")
        result = np.interp(X, self.X_, self.y_)
        if self.out_of_bounds == "nan":
            result = result.astype(float)
            result[below | above] = np.nan
        # "clip" is np.interp's default behaviour (nearest endpoint value).
        return result

    def fit_transform(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight=sample_weight)
        return self.predict(X)

    def score(self, X, y):
        """R^2 of the fitted values against ``y``."""
        y = np.asarray(y, dtype=float)
        preds = self.predict(X)
        mask = np.isfinite(preds)
        if not mask.any():
            return float("nan")
        ss_res = float(np.sum((y[mask] - preds[mask]) ** 2))
        ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
