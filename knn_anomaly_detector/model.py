"""kNN distance-based anomaly detection, from scratch with numpy only.

The anomaly score of a point is the mean distance to its ``k``
nearest neighbours in the training set. Points far from their
neighbours are unusual, so a high score means "anomalous". For
training points the point itself is excluded (it is at distance 0
from itself), so the score uses the ``k`` nearest *other* points.
The decision threshold is the ``(1 - contamination)``-quantile of the
training scores: a new point is flagged as an anomaly when its score
exceeds the threshold. ``score_samples`` returns the *negated* score
(higher means more normal), matching the scikit-learn convention, and
``predict`` returns +1 for inliers and -1 for anomalies.

Note: the threshold is calibrated on leave-one-out training scores
(each training point scored against its *other* neighbours), while
``predict`` scores a query point against the training set as an
external point. A training point therefore usually scores lower than
its calibrated value, so fewer than ``contamination`` of the training
points are flagged -- the calibrated rate applies to fresh data from
the same distribution.
"""
from __future__ import annotations

import numpy as np


def _pairwise_distances(a, b, metric):
    if metric == "euclidean":
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)
        b_sq = np.sum(b ** 2, axis=1, keepdims=True).T
        dist_sq = np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)
        return np.sqrt(dist_sq)
    if metric == "manhattan":
        return np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    raise ValueError(f"metric must be 'euclidean' or 'manhattan', got {metric!r}.")


def _validate_x(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {x.shape}.")
    if x.shape[0] == 0:
        raise ValueError("X must not be empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("X must contain only finite values.")
    return x


class KNNAnomalyDetector:
    """kNN distance-based anomaly detector.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of nearest neighbours used for the score. Must be >= 1
        and smaller than the number of training samples.
    contamination : float, default 0.1
        Expected fraction of anomalies in the training data, used to
        set the threshold as the ``(1 - contamination)``-quantile of
        the training scores. Must be in (0, 0.5].
    metric : str, default 'euclidean'
        Distance metric: ``'euclidean'`` or ``'manhattan'``.
    """

    def __init__(self, *, n_neighbors=5, contamination=0.1,
                 metric="euclidean"):
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1, got {n_neighbors!r}.")
        if not isinstance(contamination, (int, float)) or not 0 < contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination!r}.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"metric must be 'euclidean' or 'manhattan', got {metric!r}.")
        self.n_neighbors = int(n_neighbors)
        self.contamination = float(contamination)
        self.metric = metric
        self.X_fit_ = None
        self.train_scores_ = None
        self.threshold_ = None
        self.n_features_in_ = None

    def _mean_knn_distance(self, x, exclude_self=False):
        dists = _pairwise_distances(x, self.X_fit_, self.metric)
        if exclude_self:
            # The point itself is the nearest neighbour (distance 0);
            # take k + 1 neighbours and drop it.
            k = self.n_neighbors + 1
            nearest = np.partition(dists, k - 1, axis=1)[:, :k]
            return nearest[:, 1:].mean(axis=1)
        nearest = np.partition(dists, self.n_neighbors - 1, axis=1)[:, :self.n_neighbors]
        return nearest.mean(axis=1)

    def fit(self, x):
        """Fit the detector on (mostly) normal data. Returns self."""
        x = _validate_x(x)
        if len(x) <= self.n_neighbors:
            raise ValueError(
                f"Need more than n_neighbors={self.n_neighbors} training samples "
                f"to exclude each point itself, got {len(x)}.")
        self.X_fit_ = x
        self.n_features_in_ = x.shape[1]
        self.train_scores_ = self._mean_knn_distance(x, exclude_self=True)
        self.threshold_ = float(np.quantile(self.train_scores_,
                                            1.0 - self.contamination))
        return self

    def _check_fitted(self, x):
        if self.X_fit_ is None:
            raise ValueError("KNNAnomalyDetector is not fitted yet. Call fit() first.")
        x = _validate_x(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but the model was fit with {self.n_features_in_}.")
        return x

    def anomaly_score(self, x):
        """Raw anomaly score: mean distance to the k nearest neighbours.

        Higher means more anomalous.
        """
        x = self._check_fitted(x)
        return self._mean_knn_distance(x)

    def decision_function(self, x):
        """Signed distance to the threshold: > 0 means inlier."""
        return self.threshold_ - self.anomaly_score(x)

    def score_samples(self, x):
        """Anomaly score with the sklearn sign convention.

        Higher means more normal (the negated mean kNN distance).
        """
        return -self.anomaly_score(x)

    def predict(self, x):
        """Return +1 for inliers and -1 for anomalies."""
        return np.where(self.decision_function(x) >= 0, 1, -1)
