"""Bayesian linear regression with a Gaussian prior — closed-form posterior.

Model
-----
    y = X @ w + eps,   w ~ N(0, alpha^-1 I),   eps ~ N(0, beta^-1)

``alpha`` is the prior precision (1 / prior variance of each weight) and
``beta`` is the noise precision (1 / noise variance). With a Gaussian prior
and a Gaussian likelihood the posterior over the weights is Gaussian:

    S = inv(alpha * I + beta * X^T X)          (posterior covariance)
    m = beta * S @ X^T y                       (posterior mean)

Intercept handling: the intercept is handled by *centering*. During fit,
the feature means and the target mean are subtracted (``X - X_mean``,
``y - y_mean``), the posterior is computed on the centered data, and the
intercept is recovered as ``intercept_ = y_mean - X_mean @ coef_``. This
keeps the prior honest — it regularises the slopes only, not the offset —
and is equivalent to augmenting X with a column of ones while leaving that
column unpenalised. When ``fit_intercept=False`` no centering is done and
the intercept is 0.

Predictive distribution for a new point x:

    mean = x @ m (+ intercept)
    var  = 1/beta + x^T S x        (noise variance + parameter uncertainty)

so ``predict`` returns the mean and ``predict_std`` the standard deviation.
"""

import numpy as np


class BayesianRidgeRegressor:
    """Bayesian linear regression (Gaussian prior, closed-form posterior).

    Parameters
    ----------
    alpha : float, default=1.0
        Prior precision (1 / variance of the weights). Larger alpha pulls
        the weights harder towards zero.
    beta : float, default=1.0
        Noise precision (1 / noise variance). Larger beta trusts the data
        more and shrinks the predictive uncertainty.
    fit_intercept : bool, default=True
        Whether to fit an intercept (via centering, see module docstring).
    """

    def __init__(self, alpha=1.0, beta=1.0, fit_intercept=True):
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha!r}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta!r}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fit_intercept = bool(fit_intercept)

    def _validate(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be non-empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if y is not None:
            y = np.asarray(y, dtype=float)
            if y.ndim != 1:
                raise ValueError(f"y must be 1D, got shape {y.shape}")
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"X and y have mismatched lengths: "
                    f"{X.shape[0]} vs {y.shape[0]}"
                )
            if y.shape[0] == 0:
                raise ValueError("y must be non-empty")
            if not np.all(np.isfinite(y)):
                raise ValueError("y must contain only finite values")
        return X, y

    def fit(self, X, y):
        """Fit the posterior over the weights. Returns self."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha!r}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta!r}")
        X, y = self._validate(X, y)

        if self.fit_intercept:
            self._x_mean = X.mean(axis=0)
            self._y_mean = float(y.mean())
            Xc = X - self._x_mean
            yc = y - self._y_mean
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._y_mean = 0.0
            Xc, yc = X, y

        n_features = X.shape[1]
        precision = self.alpha * np.eye(n_features) + self.beta * (Xc.T @ Xc)
        self.posterior_cov_ = np.linalg.inv(precision)
        self.coef_ = self.beta * (self.posterior_cov_ @ (Xc.T @ yc))
        self.intercept_ = float(self._y_mean - self._x_mean @ self.coef_)
        self.alpha_ = self.alpha
        self.beta_ = self.beta
        self.n_features_in_ = n_features
        return self

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise ValueError("model is not fitted yet; call fit(X, y) first")

    def predict(self, X):
        """Posterior predictive mean for each row of X."""
        self._check_fitted()
        X, _ = self._validate(X)
        return X @ self.coef_ + self.intercept_

    def predict_std(self, X):
        """Posterior predictive standard deviation for each row of X.

        sqrt(1/beta + x^T S x): the first term is irreducible observation
        noise, the second is parameter uncertainty, which grows as x moves
        away from the training data.
        """
        self._check_fitted()
        X, _ = self._validate(X)
        Xc = X - self._x_mean
        quad = np.einsum("ij,jk,ik->i", Xc, self.posterior_cov_, Xc)
        return np.sqrt(1.0 / self.beta + np.maximum(quad, 0.0))

    def score(self, X, y):
        """Coefficient of determination R^2 of the predictions."""
        self._check_fitted()
        X, y = self._validate(X, y)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
