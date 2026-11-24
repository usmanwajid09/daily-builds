"""Sparse PCA: principal components with L1-sparsity via power iteration
with soft-thresholding, with deflation for multiple components.

Only numpy and the standard library are used.
"""

import numpy as np


def _soft_threshold(v, alpha):
    """Soft-thresholding operator: sign(v) * max(|v| - alpha, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - alpha, 0.0)


class SparsePCA:
    """Sparse principal component analysis.

    Each component is found by power iteration on the (deflated)
    covariance matrix, with a soft-thresholding step after every
    multiplication so that small loadings are driven to exactly zero.
    ``alpha`` controls the sparsity: ``alpha=0`` recovers (approximately)
    plain PCA, larger ``alpha`` yields sparser loadings.

    Parameters
    ----------
    n_components : int, default=2
        Number of sparse components to extract (>= 1, and at most
        ``min(n_samples, n_features)`` at fit time).
    alpha : float, default=0.5
        Sparsity strength (>= 0). Larger values zero out more loadings.
    max_iter : int, default=100
        Maximum power-iteration steps per component.
    tol : float, default=1e-6
        Convergence tolerance on ``1 - |v_new . v|``.
    random_state : int or None, default=None
        Seed for the random starting vectors (reproducible fits).

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse loading vectors, one per row, each of unit norm.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component (``v^T Cov v``).
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of the total variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Feature means subtracted before fitting.
    sparsity_ : float
        Fraction of loadings in ``components_`` that are exactly zero.
    n_features_in_ : int
        Number of features seen at fit time.
    """

    def __init__(self, n_components=2, alpha=0.5, max_iter=100, tol=1e-6,
                 random_state=None):
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError(
                f"n_components must be an integer >= 1, got {n_components!r}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol!r}")
        if random_state is not None and not isinstance(
                random_state, (int, np.integer)):
            raise ValueError(
                f"random_state must be an int or None, got {random_state!r}")
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.sparsity_ = None
        self.n_features_in_ = None

    # -- helpers ---------------------------------------------------------
    def _check_fitted(self):
        if self.components_ is None:
            raise ValueError(
                "SparsePCA is not fitted yet; call fit(X) first.")

    @staticmethod
    def _validate_X(X, name="X"):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array, got {X.ndim}D with shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} must be non-empty, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} must contain only finite values "
                             "(no NaN or inf).")
        return X

    def _power_iteration(self, cov, rng, prev=None):
        """One sparse power-iteration run on ``cov``; returns (v, lam).

        ``prev`` holds the components already found; every iterate is
        projected onto their orthogonal complement, so the deflated
        covariance (which is PSD on that complement) keeps yielding
        positive-variance directions and the components stay mutually
        orthogonal, which makes reconstruction error fall monotonically
        as components are added.
        """
        n_features = cov.shape[0]
        v = rng.standard_normal(n_features)
        if prev is not None and prev.shape[0] > 0:
            v = v - prev.T @ (prev @ v)
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(self.max_iter):
            w = cov @ v
            v_new = _soft_threshold(w, self.alpha)
            norm = np.linalg.norm(v_new)
            if norm == 0.0:
                # Thresholding killed everything: fall back to the plain
                # power step so the iteration can still make progress.
                v_new = w
                norm = np.linalg.norm(v_new)
                if norm == 0.0:
                    break
            v_new /= norm
            if prev is not None and prev.shape[0] > 0:
                v_new = v_new - prev.T @ (prev @ v_new)
                norm = np.linalg.norm(v_new)
                if norm < 1e-12:
                    break
                v_new /= norm
            lam = float(v_new @ cov @ v_new)
            if 1.0 - abs(float(v_new @ v)) < self.tol:
                v = v_new
                break
            v = v_new
        lam = float(v @ cov @ v)
        return v, lam

    # -- API -------------------------------------------------------------
    def fit(self, X):
        """Fit sparse components to ``X``. Returns self."""
        X = self._validate_X(X)
        n_samples, n_features = X.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} must be <= "
                f"min(n_samples, n_features)={min(n_samples, n_features)} "
                f"for X with shape {X.shape}")
        self.mean_ = X.mean(axis=0)
        self.n_features_in_ = n_features
        Xc = X - self.mean_
        denom = max(n_samples - 1, 1)
        cov = (Xc.T @ Xc) / denom
        total_variance = float(np.trace(cov))

        rng = np.random.default_rng(self.random_state)
        components = np.zeros((self.n_components, n_features))
        explained = np.zeros(self.n_components)
        for k in range(self.n_components):
            v, lam = self._power_iteration(cov, rng, components[:k])
            components[k] = v
            explained[k] = max(lam, 0.0)
            # Deflation: remove this component's contribution.
            cov = cov - lam * np.outer(v, v)

        self.components_ = components
        self.explained_variance_ = explained
        if total_variance > 0:
            self.explained_variance_ratio_ = explained / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        self.sparsity_ = float(np.mean(components == 0.0))
        return self

    def transform(self, X):
        """Project ``X`` onto the sparse components."""
        self._check_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected "
                f"{self.n_features_in_}")
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X_transformed):
        """Reconstruct data from its sparse-component scores."""
        self._check_fitted()
        Xt = np.asarray(X_transformed, dtype=float)
        if Xt.ndim != 2:
            raise ValueError(
                f"X_transformed must be 2D, got {Xt.ndim}D")
        if Xt.shape[1] != self.n_components:
            raise ValueError(
                f"X_transformed has {Xt.shape[1]} components, expected "
                f"{self.n_components}")
        if not np.isfinite(Xt).all():
            raise ValueError("X_transformed must contain only finite values.")
        return Xt @ self.components_ + self.mean_
