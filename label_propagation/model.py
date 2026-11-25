"""Semi-supervised label propagation on a kNN graph (numpy only).

Iterates F <- alpha * S @ F + (1 - alpha) * Y on a row-normalised Gaussian
kNN graph, hard-clamping the labelled rows back to Y each iteration.
"""

import numpy as np


class LabelPropagation:
    """Label propagation on a Gaussian-weighted kNN graph.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours used to build the graph.
    max_iter : int
        Maximum number of propagation iterations.
    tol : float
        Convergence tolerance on the max absolute change of F.
    alpha : float
        Propagation strength in (0, 1). Higher alpha trusts the graph more.
    """

    def __init__(self, n_neighbors=7, max_iter=1000, tol=1e-4, alpha=0.8):
        if not isinstance(n_neighbors, (int, np.integer)) or isinstance(n_neighbors, bool):
            raise TypeError("n_neighbors must be a positive integer")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool):
            raise TypeError("max_iter must be a positive integer")
        if max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(tol, (int, float, np.floating)):
            raise TypeError("tol must be a positive number")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if not isinstance(alpha, (int, float, np.floating)):
            raise TypeError("alpha must be a number in (0, 1)")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.n_neighbors = int(n_neighbors)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.alpha = float(alpha)
        self.classes_ = None
        self.label_distributions_ = None
        self.labels_ = None
        self.n_iter_ = None
        self._X_train = None

    # -- validation -----------------------------------------------------
    def _validate_fit_inputs(self, X, y):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise TypeError("X must be a numeric array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or inf")
        try:
            y_arr = np.asarray(y)
        except (TypeError, ValueError):
            raise TypeError("y must be an array of integer labels (-1 = unlabeled)")
        if y_arr.size == 0:
            raise ValueError("y must not be empty")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(y_arr) != len(X):
            raise ValueError(
                f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}"
            )
        if not np.issubdtype(y_arr.dtype, np.integer):
            # Allow float arrays that hold only whole numbers (e.g. -1.0).
            try:
                y_float = np.asarray(y_arr, dtype=float)
            except (TypeError, ValueError):
                raise TypeError("y must contain integer labels (-1 = unlabeled)")
            if not np.isfinite(y_float).all() or not np.all(y_float == np.floor(y_float)):
                raise TypeError("y must contain integer labels (-1 = unlabeled)")
            y_arr = y_float.astype(int)
        return X, y_arr

    # -- graph ----------------------------------------------------------
    def _build_transition_matrix(self, X):
        n = len(X)
        # Squared pairwise distances.
        sq = np.sum(X ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
        np.maximum(dist_sq, 0.0, out=dist_sq)
        np.fill_diagonal(dist_sq, np.inf)  # exclude self from neighbours
        # k nearest neighbours per point.
        kth = np.partition(dist_sq, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        sigma = float(np.mean(np.sqrt(np.maximum(kth[:, -1], 0.0))))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.median(np.sqrt(np.maximum(kth, 0.0)))) or 1.0
        W = np.zeros((n, n))
        idx = np.argsort(dist_sq, axis=1)[:, : self.n_neighbors]
        rows = np.repeat(np.arange(n), self.n_neighbors)
        cols = idx.ravel()
        W[rows, cols] = np.exp(-dist_sq[rows, cols] / (2.0 * sigma ** 2))
        W = np.maximum(W, W.T)  # symmetrise
        degrees = W.sum(axis=1)
        degrees[degrees == 0] = 1.0
        return W / degrees[:, None]

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        X, y = self._validate_fit_inputs(X, y)
        n = len(X)
        if self.n_neighbors >= n:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be smaller than "
                f"n_samples ({n})"
            )
        labelled = y != -1
        if not labelled.any():
            raise ValueError("y has no labelled points (all -1)")
        labelled_classes = np.unique(y[labelled])
        if len(labelled_classes) < 2:
            raise ValueError(
                "y must contain at least 2 distinct labelled classes, "
                f"got {labelled_classes.tolist()}"
            )
        self.classes_ = labelled_classes
        class_index = {c: i for i, c in enumerate(self.classes_)}
        n_classes = len(self.classes_)
        Y = np.zeros((n, n_classes))
        for i in np.where(labelled)[0]:
            Y[i, class_index[y[i]]] = 1.0

        S = self._build_transition_matrix(X)
        F = Y.copy()
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            F_new = self.alpha * S @ F + (1.0 - self.alpha) * Y
            F_new[labelled] = Y[labelled]  # hard clamping
            change = float(np.max(np.abs(F_new - F)))
            F = F_new
            self.n_iter_ = it
            if change < self.tol:
                break
        # Rows already sum to 1 (convex combination of one-hot rows), but
        # normalise defensively against float drift.
        row_sums = F.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        F = F / row_sums
        self.label_distributions_ = F
        self.labels_ = self.classes_[np.argmax(F, axis=1)]
        self._X_train = X.copy()
        return self

    def _check_fitted(self):
        if self.label_distributions_ is None or self._X_train is None:
            raise RuntimeError("LabelPropagation is not fitted yet; call fit first")

    def predict(self, X):
        """Predict labels.

        Transductive: for the fitted training points this returns labels_.
        For new points, each point votes from its nearest training points
        weighted by the fitted label distributions (nearest-labelled-
        neighbour vote via the graph's training embeddings).
        """
        self._check_fitted()
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise TypeError("X must be a numeric array")
        if X.size == 0:
            raise ValueError("X must not be empty")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or inf")
        if X.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted with "
                f"{self._X_train.shape[1]}"
            )
        if X.shape == self._X_train.shape and np.array_equal(X, self._X_train):
            return self.labels_.copy()
        # New points: Gaussian-weighted vote of the k nearest training points.
        sq = np.sum(X ** 2, axis=1)
        sq_tr = np.sum(self._X_train ** 2, axis=1)
        dist_sq = sq[:, None] + sq_tr[None, :] - 2.0 * X @ self._X_train.T
        np.maximum(dist_sq, 0.0, out=dist_sq)
        k = min(self.n_neighbors, len(self._X_train))
        idx = np.argsort(dist_sq, axis=1)[:, :k]
        sigma = float(np.median(np.sqrt(dist_sq))) or 1.0
        weights = np.exp(-dist_sq[np.arange(len(X))[:, None], idx] / (2.0 * sigma ** 2))
        votes = np.zeros((len(X), len(self.classes_)))
        for j in range(k):
            votes += weights[:, j : j + 1] * self.label_distributions_[idx[:, j]]
        return self.classes_[np.argmax(votes, axis=1)]
