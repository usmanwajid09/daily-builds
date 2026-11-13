"""AdaBoost with decision stumps, from scratch.

A decision stump is a one-level decision tree: pick one feature and one
threshold, and predict one label below the threshold and the other
above it. AdaBoost (Freund & Schapire, 1997) combines many such weak
learners into a strong one:

1. Start with equal weights on all training samples.
2. In each round, find the stump with the lowest *weighted* error.
3. Give that stump a vote ``alpha = 0.5 * ln((1 - err) / err)`` — the
   better the stump, the louder its vote.
4. Up-weight the samples it got wrong, down-weight the ones it got
   right, so the next stump focuses on the hard cases.
5. Stop early if a stump is perfect (error 0) or no better than chance
   (error >= 0.5); the final prediction is the sign of the weighted
   vote of all stumps.

``predict`` uses all stumps; ``staged_predict`` shows the prediction
after each round, so you can see accuracy improve as stumps are added.
``feature_importances_`` sums each stump's vote by the feature it
splits on, normalised to 1.
"""

from __future__ import annotations

import numpy as np


class DecisionStump:
    """A one-feature, one-threshold binary classifier.

    Attributes:
        feature_index: feature the stump splits on.
        threshold: split point on that feature.
        polarity: +1 means "below threshold -> -1, above -> +1";
            -1 flips the two sides.
    """

    def __init__(self):
        self.feature_index: int | None = None
        self.threshold: float | None = None
        self.polarity: int = 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return -1/+1 predictions for ``x``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.feature_index is None or self.threshold is None:
            raise RuntimeError("Stump is not fitted yet - call fit() first")
        values = x[:, self.feature_index]
        preds = np.where(values < self.threshold, -1.0, 1.0)
        return preds * self.polarity


def _best_stump(x: np.ndarray, y_signed: np.ndarray,
                weights: np.ndarray) -> tuple[DecisionStump, float]:
    """Find the stump with the lowest weighted error on ``(x, y)``.

    For each feature, every midpoint between consecutive sorted unique
    values is tried as a threshold, in both polarities.
    """
    n_samples, n_features = x.shape
    best_stump = DecisionStump()
    best_error = np.inf
    for feature in range(n_features):
        values = x[:, feature]
        unique = np.unique(values)
        if unique.size < 2:
            continue
        thresholds = (unique[:-1] + unique[1:]) / 2.0
        for threshold in thresholds:
            for polarity in (1, -1):
                preds = np.where(values < threshold, -1.0, 1.0) * polarity
                error = float(np.sum(weights[preds != y_signed]))
                if error < best_error:
                    best_error = error
                    best_stump.feature_index = feature
                    best_stump.threshold = float(threshold)
                    best_stump.polarity = polarity
    return best_stump, best_error


class AdaBoostStumps:
    """Binary classifier: AdaBoost over decision stumps.

    Args:
        n_estimators: maximum number of stumps to train.
        learning_rate: scales each stump's vote; 1.0 is standard
            AdaBoost, smaller values shrink each round's contribution.

    Attributes:
        stumps_: list of fitted :class:`DecisionStump` objects.
        stump_weights_: vote (alpha) of each stump.
        feature_importances_: per-feature importance, normalised to 1.
        n_estimators_: number of stumps actually trained.
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0):
        if isinstance(n_estimators, bool) or not isinstance(
            n_estimators, (int, np.integer)
        ):
            raise ValueError("n_estimators must be an integer")
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be at least 1")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float, np.floating)
        ):
            raise ValueError("learning_rate must be a number > 0")
        if not np.isfinite(learning_rate) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be > 0")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.stumps_: list[DecisionStump] = []
        self.stump_weights_: np.ndarray | None = None
        self.feature_importances_: np.ndarray | None = None
        self.n_estimators_: int = 0
        self._labels: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AdaBoostStumps":
        """Fit the boosted ensemble to training data.

        Labels may be any two distinct values (e.g. 0/1 or -1/+1);
        internally they are mapped to -1/+1, and :meth:`predict`
        returns the original labels.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        labels = np.unique(y)
        if labels.size != 2:
            raise ValueError(
                f"AdaBoost needs exactly two distinct labels, got {labels.tolist()}"
            )
        self._labels = labels
        y_signed = np.where(y == labels[1], 1.0, -1.0)

        n_samples, n_features = x.shape
        weights = np.full(n_samples, 1.0 / n_samples)
        stumps: list[DecisionStump] = []
        alphas: list[float] = []

        for _ in range(self.n_estimators):
            stump, error = _best_stump(x, y_signed, weights)
            if stump.feature_index is None:
                break  # no feature can split (all constant)
            error = min(max(error, 1e-10), 1.0 - 1e-10)
            if error >= 0.5 - 1e-10 and stumps:
                break  # no better than chance: further rounds cannot help
            alpha = self.learning_rate * 0.5 * np.log((1.0 - error) / error)
            preds = stump.predict(x)
            weights = weights * np.exp(-alpha * y_signed * preds)
            weights /= weights.sum()
            stumps.append(stump)
            alphas.append(alpha)
            if error <= 1e-10:
                break  # perfect stump: no errors left to reweight

        if not stumps:
            raise ValueError(
                "could not train any stump: all features are constant"
            )
        self.stumps_ = stumps
        self.stump_weights_ = np.asarray(alphas)
        self.n_estimators_ = len(stumps)
        importances = np.zeros(n_features)
        for stump, alpha in zip(stumps, alphas):
            importances[stump.feature_index] += alpha
        total = importances.sum()
        self.feature_importances_ = (
            importances / total if total > 0 else importances
        )
        return self

    def _decision_function(self, x: np.ndarray) -> np.ndarray:
        """Weighted vote of all stumps (positive -> second label)."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        total = np.zeros(x.shape[0])
        for stump, alpha in zip(self.stumps_, self.stump_weights_):
            total += alpha * stump.predict(x)
        return total

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the original labels for ``x``."""
        if not self.stumps_ or self._labels is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.feature_importances_.shape[0]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.feature_importances_.shape[0])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        signed = np.where(self._decision_function(x) >= 0.0, 1, 0)
        return self._labels[signed]

    def staged_predict(self, x: np.ndarray) -> list[np.ndarray]:
        """Predictions after each boosting round, as a list of arrays.

        Element ``i`` is the prediction using only the first ``i + 1``
        stumps, so you can watch accuracy improve round by round.
        """
        if not self.stumps_ or self._labels is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.feature_importances_.shape[0]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.feature_importances_.shape[0])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        total = np.zeros(x.shape[0])
        stages = []
        for stump, alpha in zip(self.stumps_, self.stump_weights_):
            total += alpha * stump.predict(x)
            stages.append(self._labels[np.where(total >= 0.0, 1, 0)])
        return stages

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the fraction of correct predictions on ``(x, y)``."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        preds = self.predict(x)
        if y.shape[0] != preds.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        return float(np.mean(preds == y))
