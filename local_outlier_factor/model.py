"""Local Outlier Factor (LOF) anomaly detection, from scratch (numpy only).

LOF (Breunig et al., 2000) scores each point by comparing its local density
with the local densities of its k nearest neighbours:

1. k-distance(p) = distance from p to its k-th nearest neighbour.
2. Reachability distance reach(p, o) = max(k-distance(o), dist(p, o)).
   This smooths the density estimate so points inside a dense cluster do
   not get wildly different scores from tiny distance changes.
3. Local reachability density lrd(p) = 1 / mean reachability distance
   from p to its k neighbours.
4. LOF(p) = mean over neighbours o of lrd(o) / lrd(p).

LOF ≈ 1 means the point is as dense as its neighbours (an inlier);
LOF >> 1 means the point is much sparser than its neighbours (an outlier).

Score convention (matching scikit-learn): ``score_samples`` and
``decision_function`` return the *negative* LOF, so higher = more normal
and outliers get negative scores. ``negative_outlier_factor_`` holds the
per-training-point negative LOF. ``predict`` returns 1 for inliers and
-1 for outliers, using a threshold set from ``contamination`` (the
expected fraction of outliers in the training data).
"""

import numpy as np


class LocalOutlierFactor:
    """Local Outlier Factor anomaly detector.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbours used for the local density estimate. Must be
        a positive integer smaller than the number of training samples.
    contamination : float, default=0.1
        Expected proportion of outliers in the training set, in (0, 0.5].
        Sets the decision threshold: the ``contamination`` fraction of
        training points with the lowest scores are labelled -1.

    Attributes
    ----------
    negative_outlier_factor_ : ndarray of shape (n_samples,)
        Negative LOF score of each training point (higher = more normal).
    labels_ : ndarray of shape (n_samples,)
        1 (inlier) / -1 (outlier) labels for the training points.
    threshold_ : float
        Score threshold: training points with score < threshold_ are
        outliers. Equal to the ``contamination``-quantile of the training
        scores.
    offset_ : float
        Alias for ``threshold_`` (scikit-learn style name).
    """

    def __init__(self, n_neighbors=20, contamination=0.1):
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)):
            raise TypeError("n_neighbors must be an integer")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if isinstance(contamination, bool) or not isinstance(contamination, (int, float, np.floating)):
            raise TypeError("contamination must be a number")
        if not (0 < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5]")
        self.n_neighbors = int(n_neighbors)
        self.contamination = float(contamination)
        self._is_fitted = False

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_X(X):
        if isinstance(X, (list, tuple)):
            try:
                X = np.asarray(X, dtype=float)
            except (TypeError, ValueError):
                raise TypeError("X must be a numeric array")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.dtype == object or not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must be numeric")
        if X.size == 0:
            raise ValueError("X must be non-empty")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values (no NaN or inf)")
        return np.asarray(X, dtype=float)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("LocalOutlierFactor is not fitted yet; call fit(X) first")

    # ------------------------------------------------------------------
    # core computation
    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_distances(A, B):
        """Euclidean distances between rows of A (n, d) and rows of B (m, d)."""
        sq_a = np.sum(A ** 2, axis=1)[:, None]
        sq_b = np.sum(B ** 2, axis=1)[None, :]
        sq = np.maximum(sq_a + sq_b - 2.0 * A @ B.T, 0.0)
        return np.sqrt(sq)

    def _kdist_and_lrd(self, X, ref_X, ref_kdist, ref_lrd):
        """LOF of each row of X against reference set ref_X.

        ref_kdist / ref_lrd are the k-distances and LRDs of the reference
        points. Returns (lof, neighbour_indices).
        """
        dists = self._pairwise_distances(X, ref_X)
        n_ref = ref_X.shape[0]
        k = self.n_neighbors
        # indices of the k nearest reference points (excluding self when
        # X is the training set: self-distance is 0, so it sorts first and
        # we drop it by taking k+1 and skipping column 0)
        order = np.argsort(dists, axis=1, kind="stable")
        if X is ref_X:
            neigh_idx = order[:, 1:k + 1]
        else:
            neigh_idx = order[:, :k]
        neigh_dists = np.take_along_axis(dists, neigh_idx, axis=1)
        # reachability distance: max(k-distance of neighbour, actual dist)
        neigh_kdist = ref_kdist[neigh_idx]
        reach = np.maximum(neigh_kdist, neigh_dists)
        lrd = 1.0 / np.maximum(reach.mean(axis=1), 1e-12)
        lof = (ref_lrd[neigh_idx] / lrd[:, None]).mean(axis=1)
        return lof, neigh_idx

    def _fit_reference_stats(self, X):
        """k-distance and LRD of every training point (self excluded)."""
        dists = self._pairwise_distances(X, X)
        n = X.shape[0]
        k = self.n_neighbors
        order = np.argsort(dists, axis=1, kind="stable")
        neigh_idx = order[:, 1:k + 1]
        neigh_dists = np.take_along_axis(dists, neigh_idx, axis=1)
        # k-distance of point i = distance to its k-th nearest neighbour
        kdist = neigh_dists[:, -1]
        # reachability uses the *neighbour's* k-distance
        neigh_kdist = kdist[neigh_idx]
        reach = np.maximum(neigh_kdist, neigh_dists)
        lrd = 1.0 / np.maximum(reach.mean(axis=1), 1e-12)
        return kdist, lrd

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, X):
        """Fit the detector on X and compute training LOF scores."""
        X = self._validate_X(X)
        n = X.shape[0]
        if self.n_neighbors >= n:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be smaller than "
                f"n_samples ({n})"
            )
        self.X_train_ = X
        kdist, lrd = self._fit_reference_stats(X)
        self.k_distance_ = kdist
        self.lrd_ = lrd
        lof, _ = self._kdist_and_lrd(X, X, kdist, lrd)
        self.negative_outlier_factor_ = -lof
        # Threshold: the contamination fraction of training points with
        # the lowest scores are outliers.
        self.threshold_ = float(np.quantile(self.negative_outlier_factor_,
                                             self.contamination))
        self.offset_ = self.threshold_
        self.labels_ = np.where(self.negative_outlier_factor_ < self.threshold_,
                                -1, 1)
        self._is_fitted = True
        return self

    def fit_predict(self, X):
        """Fit on X and return 1 / -1 labels for the training points."""
        self.fit(X)
        return self.labels_

    def score_samples(self, X):
        """Negative LOF of each row of X (higher = more normal)."""
        self._check_fitted()
        X = self._validate_X(X)
        lof, _ = self._kdist_and_lrd(X, self.X_train_,
                                     self.k_distance_, self.lrd_)
        return -lof

    def decision_function(self, X):
        """Negative LOF of each row of X (higher = more normal).

        Same as ``score_samples``: outliers score negative, inliers score
        near -1 (LOF ≈ 1), so the decision boundary sits at ``threshold_``.
        """
        return self.score_samples(X)

    def predict(self, X):
        """1 (inlier) / -1 (outlier) for each row of X.

        New points are scored against the fitted training set and compared
        with the threshold learned from ``contamination``.
        """
        scores = self.decision_function(X)
        return np.where(scores < self.threshold_, -1, 1)
