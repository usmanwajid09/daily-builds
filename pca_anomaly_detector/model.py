"""PCA-based anomaly detection, from scratch with numpy only.

Fit PCA on normal data, then score new points by their reconstruction
error: points the low-dimensional model cannot rebuild are likely
anomalies. The threshold defaults to mean + k * std of the training
errors (or a quantile), so it adapts to the data instead of needing a
hand-picked cutoff.
"""

import numpy as np


class PCAAnomalyDetector:
    """Anomaly detector based on PCA reconstruction error.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep. Must be >= 1 and less
        than the number of features.
    threshold : {"std", "quantile"} or float
        How the anomaly threshold is set: ``"std"`` uses
        ``mean + k * std`` of training errors, ``"quantile"`` uses a
        quantile of training errors, or pass an explicit float cutoff.
    k : float
        Multiplier for the ``"std"`` rule. Must be > 0.
    quantile : float
        Quantile for the ``"quantile"`` rule, in (0, 1).
    """

    def __init__(self, n_components=2, threshold="std", k=3.0, quantile=0.99):
        if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
            raise ValueError("n_components must be a positive integer")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if isinstance(threshold, str) and threshold not in ("std", "quantile"):
            raise ValueError('threshold must be "std", "quantile", or a float')
        if not isinstance(threshold, str) and not isinstance(
            threshold, (int, float, np.floating)
        ):
            raise ValueError('threshold must be "std", "quantile", or a float')
        if isinstance(threshold, (int, float, np.floating)) and float(threshold) < 0:
            raise ValueError("explicit threshold must be >= 0")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        self.n_components = int(n_components)
        self.threshold = threshold
        self.k = float(k)
        self.quantile = float(quantile)
        self.mean_ = None
        self.components_ = None
        self.threshold_ = None
        self.train_errors_ = None

    @staticmethod
    def _as_matrix(X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features)")
        if arr.shape[0] < 2:
            raise ValueError("at least 2 samples are required")
        if arr.shape[1] < 2:
            raise ValueError("at least 2 features are required")
        if not np.isfinite(arr).all():
            raise ValueError("X must contain only finite values")
        return arr

    def fit(self, X):
        """Fit PCA on normal data and set the anomaly threshold."""
        X = self._as_matrix(X)
        n_samples, n_features = X.shape
        if self.n_components >= n_features:
            raise ValueError(
                f"n_components={self.n_components} must be less than "
                f"n_features={n_features} (otherwise reconstruction is perfect)"
            )
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = vt[: self.n_components]
        self.train_errors_ = self._errors(X)
        if isinstance(self.threshold, str):
            if self.threshold == "std":
                self.threshold_ = float(
                    self.train_errors_.mean() + self.k * self.train_errors_.std()
                )
            else:
                self.threshold_ = float(np.quantile(self.train_errors_, self.quantile))
        else:
            self.threshold_ = float(self.threshold)
        return self

    def _require_fit(self):
        if self.components_ is None:
            raise ValueError("PCAAnomalyDetector is not fitted yet; call fit() first")

    def _errors(self, X):
        X = self._as_matrix(X)
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"expected {self.components_.shape[1]} features, got {X.shape[1]}"
            )
        projected = (X - self.mean_) @ self.components_.T
        rebuilt = projected @ self.components_ + self.mean_
        return np.mean((X - rebuilt) ** 2, axis=1)

    def score_samples(self, X):
        """Reconstruction error per sample (higher = more anomalous)."""
        self._require_fit()
        return self._errors(X)

    def is_anomaly(self, X):
        """Boolean array: True where the reconstruction error exceeds the threshold."""
        return self.score_samples(X) > self.threshold_

    def predict(self, X):
        """1 for normal samples, -1 for anomalies (scikit-learn convention)."""
        return np.where(self.is_anomaly(X), -1, 1)
