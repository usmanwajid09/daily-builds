"""Quantile regression: fit a linear model for one conditional quantile.

The model minimises the pinball loss with batch gradient descent on a
smoothed subgradient.  Features are standardised internally for stable
optimisation and the coefficients are un-standardised at the end, so
``coef_`` / ``intercept_`` live in the original feature space.

Only numpy and the standard library are used.
"""

import numpy as np

__all__ = ["QuantileRegressor", "pinball_loss"]


def pinball_loss(y_true, y_pred, quantile):
    """Mean pinball (quantile) loss.

    ``loss(u) = u * (quantile - 1{u < 0})`` with ``u = y_true - y_pred``.
    The loss is zero exactly when the prediction equals the truth, and
    penalises under-prediction by ``quantile`` per unit and
    over-prediction by ``1 - quantile`` per unit, so minimising it
    recovers the ``quantile`` conditional quantile of ``y`` given ``x``.
    """
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )
    u = y_true - y_pred
    return float(np.mean(np.where(u >= 0, quantile * u, (quantile - 1.0) * u)))


class QuantileRegressor:
    """Linear quantile regression by gradient descent on the pinball loss.

    Parameters
    ----------
    quantile : float in (0, 1), default 0.5
        The conditional quantile to fit; 0.5 is the median.
    max_iter : int, default 1000
        Maximum number of gradient-descent steps.
    learning_rate : float, default 0.01
        Step size for gradient descent (must be > 0).
    tol : float, default 1e-6
        Stop early when the change in pinball loss between steps drops
        below this value.
    fit_intercept : bool, default True
        Whether to fit an intercept term.
    random_state : int or None, default None
        Seed for the small random initial weights; fixes the result
        when given.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients in the original feature space.
    intercept_ : float
        Intercept in the original feature space (0.0 if fit_intercept
        is False).
    n_iter_ : int
        Number of gradient-descent steps actually taken.
    """

    def __init__(self, quantile=0.5, max_iter=1000, learning_rate=0.01,
                 tol=1e-6, fit_intercept=True, random_state=None):
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
        if int(max_iter) <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter!r}")
        if float(learning_rate) <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate!r}")
        if float(tol) < 0:
            raise ValueError(f"tol must be non-negative, got {tol!r}")
        if random_state is not None and not isinstance(
                random_state, (int, np.integer)):
            raise ValueError(
                f"random_state must be an int or None, got {random_state!r}")
        self.quantile = float(quantile)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.tol = float(tol)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    # -- validation -----------------------------------------------------
    @staticmethod
    def _validate_xy(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("X and y must be non-empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: {X.shape[0]} != {y.shape[0]}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        return X, y

    def _check_fitted(self):
        if self.coef_ is None:
            raise ValueError("QuantileRegressor is not fitted yet; call fit() first")

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        """Fit the linear model for ``self.quantile`` by gradient descent.

        The pinball loss is piecewise linear, so its subgradient is a
        step function: ``quantile - 1{y < pred}`` per sample.  A smoothed
        version replaces the indicator with a sigmoid of the residual
        scaled by a small bandwidth (5% of the target's standard
        deviation), which keeps the descent stable while converging to
        the same optimum.  The gradient of the loss w.r.t. the weights
        is the negative of that term times the (standardised) features,
        so each step adds ``learning_rate * (quantile - 1{y < pred})``
        along the features — the sign the subgradient formula implies.
        """
        X, y = self._validate_xy(X, y)
        n, n_features = X.shape

        # Standardise features internally for stable optimisation.
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std = np.where(self._x_std > 0, self._x_std, 1.0)
        Xs = (X - self._x_mean) / self._x_std

        rng = np.random.default_rng(self.random_state)
        w = rng.normal(scale=0.01, size=n_features)
        b = 0.0

        # Smoothing bandwidth for the indicator: 5% of y's spread.
        h = max(0.05 * float(np.std(y)), 1e-8)

        prev_loss = np.inf
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            pred = Xs @ w + b
            resid = y - pred
            # Smoothed subgradient term: tau - 1{y < pred}, with the
            # indicator softened to sigmoid((pred - y) / h).
            s = self.quantile - 1.0 / (1.0 + np.exp(-(pred - y) / h))
            # d loss / d w = -(1/n) * Xs^T s ;  d loss / d b = -(1/n) * sum(s)
            grad_w = -(Xs.T @ s) / n
            w -= self.learning_rate * grad_w
            if self.fit_intercept:
                b -= self.learning_rate * (-float(np.mean(s)))
            loss = pinball_loss(y, Xs @ w + b, self.quantile)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.n_iter_ = n_iter
        # Un-standardise: pred = Xs @ w + b = X @ (w / std) + (b - mean/std @ w)
        self.coef_ = w / self._x_std
        self.intercept_ = float(b - np.dot(self._x_mean / self._x_std, w)) \
            if self.fit_intercept else 0.0
        return self

    # -- prediction / scoring -------------------------------------------
    def predict(self, X):
        """Predict the fitted conditional quantile for X."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted with "
                f"{self.coef_.shape[0]}")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R^2 of the predictions, for comparability with other regressors."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
