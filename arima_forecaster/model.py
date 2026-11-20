"""Autoregressive AR(p) time-series forecaster, fit by ordinary least squares.

The model is

    y[t] = intercept + phi_1 * y[t-1] + ... + phi_p * y[t-p] + eps[t],

with eps[t] white noise of variance sigma2.  Only NumPy is used.
"""

import math

import numpy as np

__all__ = ["ARForecaster"]


def _normal_quantile(probability: float) -> float:
    """Inverse CDF of the standard normal distribution (Acklam's method).

    Accurate to ~1.15e-9 absolute error; no SciPy needed.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if probability < p_low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if probability <= p_high:
        q = probability - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


# Two-sided critical values for the most common significance levels.
_Z_TABLE = {0.01: 2.5758293035489004, 0.05: 1.959963984540054, 0.10: 1.6448536269514722}


class ARForecaster:
    """AR(p) forecaster fit by ordinary least squares.

    Parameters
    ----------
    p : int
        Autoregressive order (number of lags). Must be a positive integer.

    Attributes
    ----------
    coef_ : np.ndarray, shape (p,)
        Fitted AR coefficients (phi_1, ..., phi_p). Available after ``fit``.
    intercept_ : float
        Fitted intercept. Available after ``fit``.
    sigma2_ : float
        Residual variance estimated from in-sample residuals.
        Available after ``fit``.
    fitted : bool
        True once :meth:`fit` has been called successfully.
    """

    def __init__(self, p: int = 2):
        if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
            raise ValueError(f"p must be a positive integer, got {p!r}.")
        if int(p) < 1:
            raise ValueError(f"p must be a positive integer, got {p}.")
        self.p = int(p)
        self.coef_ = None
        self.intercept_ = None
        self.sigma2_ = None
        self.fitted = False
        self._last_values = None

    def fit(self, series):
        """Fit the AR(p) model to a 1D series by ordinary least squares.

        Parameters
        ----------
        series : array-like of floats, shape (n,)
            Observed time series. Must be 1D, finite, and longer than ``p``.

        Returns
        -------
        self
        """
        try:
            y = np.asarray(series, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"series must be a 1D array-like of numbers, got {series!r}."
            ) from exc
        if y.ndim != 1:
            raise ValueError(
                f"series must be 1D, got array with shape {y.shape}."
            )
        if y.size == 0:
            raise ValueError("series must not be empty.")
        if not np.all(np.isfinite(y)):
            raise ValueError("series must contain only finite values (no NaN or inf).")
        if y.size <= self.p:
            raise ValueError(
                f"series length ({y.size}) must be greater than p ({self.p})."
            )

        n = y.size
        # Lagged design matrix: rows [1, y[t-1], ..., y[t-p]], target y[t].
        x = np.column_stack(
            [np.ones(n - self.p)]
            + [y[self.p - i - 1: n - i - 1] for i in range(self.p)]
        )
        target = y[self.p:]
        beta, *_ = np.linalg.lstsq(x, target, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = np.asarray(beta[1:], dtype=float)
        residuals = target - x @ beta
        dof = max(n - self.p - (self.p + 1), 1)
        self.sigma2_ = float(np.sum(residuals ** 2) / dof)
        self._last_values = y[-self.p:].copy()
        self.fitted = True
        return self

    def _check_steps(self, steps):
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
            raise ValueError(f"steps must be a positive integer, got {steps!r}.")
        if int(steps) < 1:
            raise ValueError(f"steps must be a positive integer, got {steps}.")
        return int(steps)

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(
                "ARForecaster is not fitted yet. Call fit(series) first."
            )

    def predict(self, steps: int = 1) -> np.ndarray:
        """Forecast ``steps`` values ahead, recursively.

        Parameters
        ----------
        steps : int
            Number of steps to forecast. Must be a positive integer.

        Returns
        -------
        np.ndarray, shape (steps,)
            Point forecasts.
        """
        steps = self._check_steps(steps)
        self._check_fitted()
        history = list(self._last_values)  # oldest -> newest
        out = np.empty(steps, dtype=float)
        for h in range(steps):
            value = self.intercept_ + float(
                np.dot(self.coef_, history[::-1])
            )
            out[h] = value
            history.append(value)
            history.pop(0)
        return out

    def _psi_weights(self, steps: int) -> np.ndarray:
        """MA (psi) weights of the AR process, psi_0 = 1.

        psi_j = sum_{i=1}^{min(j, p)} phi_i * psi_{j-i}.
        """
        psi = np.zeros(steps, dtype=float)
        psi[0] = 1.0
        for j in range(1, steps):
            total = 0.0
            for i in range(1, min(j, self.p) + 1):
                total += self.coef_[i - 1] * psi[j - i]
            psi[j] = total
        return psi

    def forecast_interval(self, steps: int = 1, alpha: float = 0.05):
        """Forecast with a two-sided (1 - alpha) prediction interval.

        Interval half-widths grow with the horizon: the forecast-error
        variance at horizon h is sigma2 * sum_{j=0}^{h-1} psi_j**2, where
        psi are the MA weights of the fitted AR process.

        Parameters
        ----------
        steps : int
            Number of steps to forecast. Must be a positive integer.
        alpha : float
            Significance level in (0, 1); e.g. 0.05 gives a 95% interval.
            Exact table values are used for 0.01, 0.05 and 0.10; other
            values use an inverse-normal approximation.

        Returns
        -------
        (mean, lower, upper) : tuple of np.ndarray, each shape (steps,)
        """
        steps = self._check_steps(steps)
        self._check_fitted()
        if not isinstance(alpha, (int, float, np.floating)) or isinstance(alpha, bool):
            raise ValueError(f"alpha must be a number in (0, 1), got {alpha!r}.")
        alpha = float(alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

        z = _Z_TABLE.get(alpha)
        if z is None:
            z = _normal_quantile(1.0 - alpha / 2.0)

        mean = self.predict(steps)
        psi = self._psi_weights(steps)
        # Cumulative sum of squared psi weights -> error variance per horizon.
        cum_sq = np.cumsum(psi ** 2)
        std = math.sqrt(max(self.sigma2_, 0.0)) * np.sqrt(cum_sq)
        lower = mean - z * std
        upper = mean + z * std
        return mean, lower, upper
