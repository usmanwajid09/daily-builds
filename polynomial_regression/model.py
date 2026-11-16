"""Polynomial regression with ridge regularisation.

Single-feature polynomial regression: expands ``x`` into
``[1, x, x**2, ..., x**degree]`` and fits a ridge-regularised linear model
with the closed-form solution ``(X^T X + alpha * I)^-1 X^T y``, leaving the
intercept unpenalised.  The input is standardised inside :meth:`fit`
(train mean/std) so higher degrees stay numerically stable; the fitted
coefficients are converted back to the original ``x`` basis, so
``coef_[0]`` is the intercept and ``coef_[k]`` multiplies ``x**k``.
"""

import math

import numpy as np

__all__ = ["PolynomialRegression", "select_degree"]


def _validate_degree(degree):
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise ValueError(f"degree must be a positive integer, got {degree!r}")
    if degree < 1:
        raise ValueError(f"degree must be a positive integer, got {degree}")


def _validate_alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise ValueError(f"alpha must be a non-negative number, got {alpha!r}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")


def _as_1d_feature(X, name="X"):
    """Return a 1-D float array for a single-feature input."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a single feature: shape (n,) or (n, 1), got {arr.shape}"
        )
    return arr


def _validate_xy(X, y):
    x = _as_1d_feature(X, "X")
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-D with shape (n,), got {y_arr.shape}")
    if x.size == 0 or y_arr.size == 0:
        raise ValueError("X and y must be non-empty")
    if x.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y have mismatched lengths: {x.shape[0]} vs {y_arr.shape[0]}"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("X and y must contain only finite values")
    return x, y_arr


class PolynomialRegression:
    """Polynomial regression fitted by the closed-form ridge solution.

    Parameters
    ----------
    degree : int, default 2
        Highest polynomial power. Must be a positive integer.
    alpha : float, default 1.0
        Ridge penalty strength. Must be non-negative; the intercept is
        never penalised.

    Attributes
    ----------
    coef_ : ndarray, shape (degree + 1,)
        Fitted coefficients in the original ``x`` basis; ``coef_[0]`` is
        the intercept and ``coef_[k]`` multiplies ``x**k``.
    degree, alpha : as passed to the constructor.
    x_mean_, x_std_ : float
        Train mean/std used to standardise ``x`` inside the fit.
    """

    def __init__(self, degree=2, alpha=1.0):
        _validate_degree(degree)
        _validate_alpha(alpha)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.coef_ = None
        self.x_mean_ = None
        self.x_std_ = None

    def _design(self, x):
        """Polynomial features of the standardised input, with intercept."""
        z = (x - self.x_mean_) / self.x_std_
        return np.vander(z, N=self.degree + 1, increasing=True)

    @staticmethod
    def _to_raw_coef(std_coef, mean, std):
        """Convert coefficients from the standardised basis back to raw x.

        z = (x - mean) / std, so z**k expands by the binomial theorem into
        powers of x; accumulate each term's contribution.
        """
        degree = len(std_coef) - 1
        raw = np.zeros(degree + 1)
        for k in range(degree + 1):
            for j in range(k + 1):
                raw[j] += (
                    std_coef[k]
                    * math.comb(k, j)
                    * (-mean) ** (k - j)
                    / std**k
                )
        return raw

    def fit(self, X, y):
        """Fit the model and return self."""
        x, y_arr = _validate_xy(X, y)
        self.x_mean_ = float(np.mean(x))
        self.x_std_ = float(np.std(x))
        if self.x_std_ == 0.0:
            raise ValueError("X has zero variance; cannot standardise a constant feature")

        design = self._design(x)
        penalty = self.alpha * np.eye(self.degree + 1)
        penalty[0, 0] = 0.0  # intercept is not penalised
        std_coef = np.linalg.solve(
            design.T @ design + penalty, design.T @ y_arr
        )
        self.coef_ = self._to_raw_coef(std_coef, self.x_mean_, self.x_std_)
        return self

    def predict(self, X):
        """Predict target values for X. Returns an array of shape (n,)."""
        if self.coef_ is None:
            raise RuntimeError("predict called before fit; call fit(X, y) first")
        x = _as_1d_feature(X, "X")
        if x.size == 0:
            raise ValueError("X must be non-empty")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must contain only finite values")
        # Predict from the stored raw-basis coefficients.
        powers = np.vander(x, N=self.degree + 1, increasing=True)
        return powers @ self.coef_

    def score(self, X, y):
        """R^2 of the model on (X, y)."""
        x, y_arr = _validate_xy(X, y)
        y_pred = self.predict(x)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true, dtype=float) - y_pred) ** 2))


def select_degree(X_train, y_train, X_val, y_val, degrees, alpha=1.0):
    """Pick the best polynomial degree by validation MSE.

    Fits a :class:`PolynomialRegression` for each degree in ``degrees``
    and returns ``(best_degree, best_mse, results)`` where ``results``
    maps each degree to its validation MSE.
    """
    _validate_alpha(alpha)
    degrees = list(degrees)
    if not degrees:
        raise ValueError("degrees must be a non-empty iterable of positive integers")
    results = {}
    for degree in degrees:
        _validate_degree(degree)
        model = PolynomialRegression(degree=int(degree), alpha=alpha)
        model.fit(X_train, y_train)
        results[int(degree)] = _mse(y_val, model.predict(X_val))
    best_degree = min(results, key=results.get)
    return best_degree, results[best_degree], results
