"""Bayesian logistic regression with a Laplace approximation, from scratch.

The model is ``p(y = 1 | x) = sigmoid(w^T x)`` with a Gaussian prior
``w ~ N(0, alpha^-1 I)``. The posterior has no closed form, so it is
approximated by a Gaussian centred at the MAP estimate (found by
Newton-Raphson / IRLS) with covariance equal to the inverse Hessian of
the negative log posterior at the MAP:

    S = (X^T W X + A)^-1,

where ``W`` is the diagonal matrix of ``p (1 - p)`` at the MAP and
``A`` is the prior precision (the intercept is not penalised).
Predictive probabilities use MacKay's probit approximation, which
folds the posterior uncertainty into the prediction:

    p(y = 1 | x) ~= sigmoid(kappa * m^T x),
    kappa = (1 + pi / 8 * x^T S x)^(-1/2).

Points far from the training data get ``x^T S x`` large, ``kappa``
small, and predictions pulled towards 0.5 -- the model knows what it
does not know.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _validate_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent lengths: {x.shape[0]} vs {y.shape[0]}.")
    if x.shape[0] == 0:
        raise ValueError("X and y must not be empty.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1 for binary logistic regression.")
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes 0 and 1.")
    return x, y


class BayesianLogisticRegression:
    """Bayesian logistic regression via the Laplace approximation.

    Parameters
    ----------
    alpha : float, default 1.0
        Prior precision of the weights (``w ~ N(0, alpha^-1 I)``).
        Must be > 0. Larger values shrink the weights more strongly.
    fit_intercept : bool, default True
        Whether to fit an intercept (which is not penalised).
    max_iter : int, default 100
        Maximum Newton-Raphson iterations. Must be >= 1.
    tol : float, default 1e-6
        Convergence tolerance on the weight update. Must be > 0.
    """

    def __init__(self, *, alpha=1.0, fit_intercept=True, max_iter=100,
                 tol=1e-6):
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha!r}.")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError(f"max_iter must be an integer >= 1, got {max_iter!r}.")
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol!r}.")
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.posterior_mean_ = None
        self.posterior_cov_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.n_iter_ = None

    def _design(self, x):
        if self.fit_intercept:
            return np.column_stack([np.ones(len(x)), x])
        return x

    def _prior_precision(self, n_params):
        a = np.full(n_params, self.alpha)
        if self.fit_intercept:
            a[0] = 1e-8  # intercept is effectively unpenalised
        return np.diag(a)

    def fit(self, x, y):
        """Fit the MAP estimate and Laplace posterior. Returns self."""
        x, y = _validate_xy(x, y)
        n, p = x.shape
        self.n_features_in_ = p
        x_design = self._design(x)
        n_params = x_design.shape[1]
        prior_prec = self._prior_precision(n_params)
        w = np.zeros(n_params)
        for it in range(1, self.max_iter + 1):
            p_hat = _sigmoid(x_design @ w)
            grad = x_design.T @ (y - p_hat) - prior_prec @ w
            w_mat = p_hat * (1.0 - p_hat)
            hess = (x_design * w_mat[:, None]).T @ x_design + prior_prec
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hess) @ grad
            w = w + step
            if float(np.max(np.abs(step))) < self.tol:
                break
        self.n_iter_ = it
        self.posterior_mean_ = w
        p_hat = _sigmoid(x_design @ w)
        w_mat = p_hat * (1.0 - p_hat)
        hess = (x_design * w_mat[:, None]).T @ x_design + prior_prec
        try:
            cov = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(hess)
        self.posterior_cov_ = (cov + cov.T) / 2.0
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = w.copy()
        return self

    def _linear_predictor(self, x):
        if self.posterior_mean_ is None:
            raise ValueError("BayesianLogisticRegression is not fitted yet. Call fit() first.")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {x.shape}.")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but the model was fit with {self.n_features_in_}.")
        if x.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must contain only finite values.")
        x_design = self._design(x)
        mean = x_design @ self.posterior_mean_
        var = np.einsum("ij,jk,ik->i", x_design, self.posterior_cov_, x_design)
        return mean, np.maximum(var, 0.0)

    def predict_proba(self, x, *, use_uncertainty=True):
        """Predict class probabilities.

        With ``use_uncertainty=True`` (default) the Laplace posterior
        uncertainty is folded in via MacKay's probit approximation, so
        points far from the data get probabilities near 0.5. With
        ``use_uncertainty=False`` the plug-in MAP probabilities are
        returned. Returns an ``(n, 2)`` array of ``[P(y=0), P(y=1)]``.
        """
        mean, var = self._linear_predictor(x)
        if use_uncertainty:
            kappa = (1.0 + (np.pi / 8.0) * var) ** (-0.5)
            p1 = _sigmoid(kappa * mean)
        else:
            p1 = _sigmoid(mean)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x):
        """Predict class labels (0 or 1)."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def score(self, x, y):
        """Return the classification accuracy."""
        x, y = _validate_xy(x, y)
        return float(np.mean(self.predict(x) == y))
