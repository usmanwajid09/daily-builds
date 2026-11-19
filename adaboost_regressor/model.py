"""AdaBoost regression, from scratch with numpy only.

An ensemble of decision stumps (depth-1 trees) trained sequentially: at
each stage a stump is fit to the sample-weighted data, its weighted error
is measured with the linear loss (absolute error scaled by the worst
error in the stage), and sample weights are boosted for the points the
stump got wrong. The final prediction is the weighted median of the
stumps' predictions, which is robust to a few badly wrong stumps.
"""
from __future__ import annotations

import numpy as np


class _Stump:
    """A depth-1 regression tree (one split, two leaf values)."""

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
        self.n_features_in_ = None

    def fit(self, x, y, sample_weight):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        w = np.asarray(sample_weight, dtype=float).ravel()
        self.n_features_in_ = x.shape[1]
        total_w = w.sum()
        best = None  # (weighted mse, feature, threshold, left, right)
        for feature in range(x.shape[1]):
            order = np.argsort(x[:, feature], kind="stable")
            xs = x[order, feature]
            ys = y[order]
            ws = w[order]
            distinct = xs[1:] != xs[:-1]
            if not np.any(distinct):
                continue
            idx = np.flatnonzero(distinct)
            cum_w = np.cumsum(ws)
            cum_wy = np.cumsum(ws * ys)
            cum_wy2 = np.cumsum(ws * ys * ys)
            w_left = cum_w[idx]
            w_right = total_w - w_left
            if np.any(w_left <= 0) or np.any(w_right <= 0):
                continue
            # Weighted MSE of each side: sum(w*y^2) - (sum(w*y))^2 / sum(w).
            mse_left = cum_wy2[idx] - cum_wy[idx] ** 2 / w_left
            mse_right = (cum_wy2[-1] - cum_wy2[idx]) \
                - (cum_wy[-1] - cum_wy[idx]) ** 2 / w_right
            mse = mse_left + mse_right
            j = int(np.argmin(mse))
            if best is None or mse[j] < best[0]:
                left_value = cum_wy[idx[j]] / w_left[j]
                right_value = (cum_wy[-1] - cum_wy[idx[j]]) / w_right[j]
                best = (float(mse[j]), feature,
                        float((xs[idx[j]] + xs[idx[j] + 1]) / 2.0),
                        float(left_value), float(right_value))
        if best is None:
            # No split possible: constant prediction.
            self.feature = 0
            self.threshold = np.inf
            value = float(np.sum(w * y) / total_w)
            self.left_value = value
            self.right_value = value
        else:
            _, self.feature, self.threshold, self.left_value, \
                self.right_value = best
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features, expected "
                             f"{self.n_features_in_}")
        return np.where(x[:, self.feature] <= self.threshold,
                        self.left_value, self.right_value)


def _weighted_median(values, weights):
    """Weighted median of ``values`` with ``weights``.

    Weights are the stump weights (log(1/beta)); a stump whose error
    exceeded 0.5 gets a negative weight, which the median handles by
    treating it as a vote against that stump's prediction. If the weights
    sum to zero or less, fall back to the plain (unweighted) median.
    """
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype=float)[order]
    total = weights.sum()
    if total <= 0:
        return float(np.median(values))
    cum = np.cumsum(weights)
    return float(values[np.searchsorted(cum, 0.5 * total)])


class AdaBoostRegressor:
    """AdaBoost regression over decision stumps (linear loss).

    Parameters
    ----------
    n_estimators : int
        Number of stumps in the ensemble.
    learning_rate : float
        Shrinkage applied to each stump's weight.
    loss : str
        Loss used to measure each stump's error: ``"linear"`` (absolute
        error scaled by the worst error) or ``"square"`` (squared scaled
        error).
    random_state : int or None
        Accepted for API symmetry; the fit is deterministic.
    """

    def __init__(self, *, n_estimators=50, learning_rate=1.0,
                 loss="linear", random_state=None):
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer, "
                             f"got {n_estimators!r}")
        if not (0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1], "
                             f"got {learning_rate!r}")
        if loss not in ("linear", "square"):
            raise ValueError('loss must be "linear" or "square", '
                             f"got {loss!r}")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.random_state = random_state

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if len(x) != len(y):
            raise ValueError("X and y have different lengths "
                             f"({len(x)} vs {len(y)})")
        if len(x) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        self.n_features_in_ = x.shape[1]
        n = len(y)

        weights = np.full(n, 1.0 / n)
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.feature_importances_ = np.zeros(self.n_features_in_)

        for _ in range(self.n_estimators):
            stump = _Stump().fit(x, y, weights)
            pred = stump.predict(x)
            abs_err = np.abs(y - pred)
            max_err = abs_err.max()
            if max_err == 0:
                # Perfect fit: keep this stump, stop adding more.
                self.estimators_.append(stump)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(0.0)
                break
            if self.loss == "linear":
                loss = abs_err / max_err
            else:
                loss = (abs_err / max_err) ** 2
            error = float(np.sum(weights * loss))
            if error >= 1.0:
                # No better than predicting the worst point everywhere:
                # drop this stump and stop.
                break
            error = max(error, 1e-12)
            beta = error / (1.0 - error)
            stump_weight = self.learning_rate * np.log(1.0 / beta)
            self.estimators_.append(stump)
            self.estimator_weights_.append(stump_weight)
            self.estimator_errors_.append(error)
            self.feature_importances_[stump.feature] += stump_weight
            # Boost weights of the points this stump got wrong.
            weights = weights * beta ** (1.0 - loss)
            weights = weights / weights.sum()

        if not self.estimators_:
            raise ValueError("Could not fit any estimator — the first stump "
                             "had error >= 1.0")
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        self.estimator_weights_ = np.array(self.estimator_weights_)
        self.estimator_errors_ = np.array(self.estimator_errors_)
        return self

    def _staged_predictions(self, x):
        """Predictions of each stump, shape (n_estimators, n_samples)."""
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features, expected "
                             f"{self.n_features_in_}")
        return np.array([stump.predict(x) for stump in self.estimators_])

    def predict(self, x):
        preds = self._staged_predictions(x)
        out = np.empty(preds.shape[1])
        for i in range(preds.shape[1]):
            out[i] = _weighted_median(preds[:, i], self.estimator_weights_)
        return out

    def staged_predict(self, x):
        """Yield predictions using the first k stumps, k = 1..n_estimators."""
        self._check_fitted()
        preds = self._staged_predictions(x)
        for k in range(1, len(self.estimators_) + 1):
            out = np.empty(preds.shape[1])
            weights = self.estimator_weights_[:k]
            for i in range(preds.shape[1]):
                out[i] = _weighted_median(preds[:k, i], weights)
            yield out

    def score(self, x, y):
        """Return the R^2 of the predictions on (x, y)."""
        y = np.asarray(y, dtype=float).ravel()
        pred = self.predict(x)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    def _check_fitted(self):
        if not hasattr(self, "estimators_"):
            raise ValueError("Model is not fitted yet — call fit() first")
