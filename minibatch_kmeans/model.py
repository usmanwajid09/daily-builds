"""MiniBatch K-Means clustering from scratch, numpy only."""

import numpy as np


class MiniBatchKMeans:
    """Mini-batch K-Means clustering.

    Like K-Means, but each step updates the centres from a small random
    mini-batch instead of the whole dataset, so it scales to data that
    does not fit comfortably in memory and converges in far fewer full
    passes. Each centre keeps a running count of the points assigned to
    it; the learning rate for a centre is 1 / count, so early updates
    move fast and later ones settle.

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters. Must be positive.
    batch_size : int, default 100
        Size of each mini-batch. Must be positive.
    max_iter : int, default 100
        Maximum number of mini-batch steps. Must be positive.
    init : {"k-means++", "random"}, default "k-means++"
        How to choose the initial centres.
    random_state : int or None, default None
        Seed for centre initialisation and mini-batch sampling.
    tol : float, default 1e-4
        Stop early when the relative change in inertia between checks
        is below this. Set to 0 to disable.
    """

    def __init__(self, n_clusters=8, batch_size=100, max_iter=100,
                 init="k-means++", random_state=None, tol=1e-4):
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter!r}")
        if init not in ("k-means++", "random"):
            raise ValueError(f"init must be 'k-means++' or 'random', got {init!r}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol!r}")
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.tol = tol

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values")
        return X

    def _init_centers(self, X, rng):
        n_samples = X.shape[0]
        if self.init == "random":
            idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[idx].copy()
        # k-means++: first centre uniform, the rest with probability
        # proportional to the squared distance to the nearest centre.
        centers = np.empty((self.n_clusters, X.shape[1]))
        first = rng.integers(n_samples)
        centers[0] = X[first]
        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for k in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                idx = rng.integers(n_samples)
            else:
                probs = closest_sq / total
                idx = rng.choice(n_samples, p=probs)
            centers[k] = X[idx]
            closest_sq = np.minimum(
                closest_sq, np.sum((X - centers[k]) ** 2, axis=1))
        return centers

    def _assign(self, X):
        dist_sq = np.sum((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
                         axis=2)
        labels = np.argmin(dist_sq, axis=1)
        return labels, np.min(dist_sq, axis=1)

    def _update_batch(self, X_batch):
        labels, _ = self._assign(X_batch)
        for k in range(self.n_clusters):
            members = X_batch[labels == k]
            if len(members) == 0:
                continue
            # Running mean update, one point at a time, with the
            # per-centre count as the learning rate.
            for point in members:
                self.counts_[k] += 1
                eta = 1.0 / self.counts_[k]
                self.cluster_centers_[k] += eta * (point - self.cluster_centers_[k])

    # -- API -------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        X = self._check_X(X)
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) exceeds the number of "
                f"samples ({X.shape[0]})")
        rng = np.random.default_rng(self.random_state)
        self.cluster_centers_ = self._init_centers(X, rng)
        self.counts_ = np.zeros(self.n_clusters, dtype=np.int64)
        n_samples = X.shape[0]
        prev_inertia = None
        self.n_iter_ = 0
        for it in range(self.max_iter):
            self.n_iter_ = it + 1
            idx = rng.integers(0, n_samples,
                               size=min(self.batch_size, n_samples))
            self._update_batch(X[idx])
            if self.tol > 0 and (it + 1) % 10 == 0:
                inertia = self.inertia(X)
                if prev_inertia is not None and prev_inertia > 0:
                    if abs(prev_inertia - inertia) / prev_inertia < self.tol:
                        break
                prev_inertia = inertia
        labels, dist_sq = self._assign(X)
        self.labels_ = labels
        self.inertia_ = float(dist_sq.sum())
        return self

    def partial_fit(self, X, y=None):
        """Update the centres from one mini-batch (online learning).

        The model must already be initialised: call ``fit`` first, or
        call ``partial_fit`` once on a batch with at least
        ``n_clusters`` samples and it will initialise from that batch.
        """
        X = self._check_X(X)
        if not hasattr(self, "cluster_centers_"):
            if X.shape[0] < self.n_clusters:
                raise ValueError(
                    f"First partial_fit needs at least n_clusters "
                    f"({self.n_clusters}) samples, got {X.shape[0]}")
            rng = np.random.default_rng(self.random_state)
            self.cluster_centers_ = self._init_centers(X, rng)
            self.counts_ = np.zeros(self.n_clusters, dtype=np.int64)
        self._update_batch(X)
        labels, dist_sq = self._assign(X)
        self.inertia_ = float(dist_sq.sum())
        return self

    def predict(self, X):
        """Assign each sample to the nearest centre."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError(
                "MiniBatchKMeans is not fitted yet; call fit() first")
        X = self._check_X(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected "
                f"{self.cluster_centers_.shape[1]}")
        labels, _ = self._assign(X)
        return labels

    def fit_predict(self, X, y=None):
        """Fit the model and return the cluster labels for X."""
        return self.fit(X).labels_

    def transform(self, X):
        """Distances from each sample to each centre."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError(
                "MiniBatchKMeans is not fitted yet; call fit() first")
        X = self._check_X(X)
        return np.sqrt(np.sum(
            (X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2))

    def inertia(self, X):
        """Sum of squared distances to the nearest centre."""
        X = self._check_X(X)
        _, dist_sq = self._assign(X)
        return float(dist_sq.sum())
