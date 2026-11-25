"""FastICA: blind source separation via the fixed-point algorithm.

Separates a linear mixture X = S @ A.T of independent non-Gaussian sources
into its independent components. Only numpy is used.

Pipeline
--------
1. Centre X (subtract the per-feature mean).
2. Whiten the centred data with the eigendecomposition of its covariance,
   so the whitened data has identity covariance.
3. Run the fixed-point iteration with the tanh nonlinearity
   (g(u) = tanh(u), g'(u) = 1 - tanh(u)^2) and symmetric decorrelation
   W <- (W W^T)^(-1/2) W after every iteration, until the unmixing matrix
   stops changing (max |diag(W_new @ W_old.T)| close to 1).

Only non-Gaussian sources are identifiable: a Gaussian source has no
direction of maximal non-Gaussianity for the algorithm to find.
"""

import numpy as np


class FastICA:
    """FastICA blind source separation (symmetric, tanh nonlinearity).

    Parameters
    ----------
    n_components : int or None
        Number of independent components to extract. Defaults to the
        number of features.
    max_iter : int
        Maximum number of fixed-point iterations.
    tol : float
        Convergence tolerance on ``1 - max |diag(W_new @ W_old.T)|``.
    random_state : int or None
        Seed for the random initial unmixing matrix. Same seed gives the
        same result (up to component order/sign, which the iteration may
        still permute between runs with different seeds).
    """

    def __init__(self, n_components=None, max_iter=200, tol=1e-4,
                 random_state=None):
        if n_components is not None:
            if isinstance(n_components, bool) or not isinstance(
                    n_components, (int, np.integer)):
                raise TypeError(
                    f"n_components must be an int or None, got "
                    f"{type(n_components).__name__}")
            if n_components <= 0:
                raise ValueError(
                    f"n_components must be a positive int, got {n_components}")
        if max_iter is not None and (
                isinstance(max_iter, bool) or not isinstance(
                    max_iter, (int, np.integer)) or max_iter <= 0):
            raise ValueError("max_iter must be a positive int")
        if tol is not None and (not np.isfinite(tol) or tol <= 0):
            raise ValueError("tol must be a positive finite number")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        # Fitted attributes (set by fit).
        self.components_ = None
        self.mixing_ = None
        self.mean_ = None
        self.whitening_ = None
        self.n_iter_ = None

    # -- validation -----------------------------------------------------
    @staticmethod
    def _validate_X(X):
        if isinstance(X, (list, tuple)):
            try:
                X = np.asarray(X, dtype=float)
            except (TypeError, ValueError):
                raise TypeError("X must be a numeric array")
        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"X must be a numpy array, got {type(X).__name__}")
        if X.dtype.kind not in "fiu":
            raise TypeError(
                f"X must be numeric, got dtype {X.dtype}")
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_samples, n_features), got {X.ndim}D "
                f"with shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                f"X must be non-empty, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or inf")
        return X.astype(float, copy=False)

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError(
                "FastICA is not fitted yet; call fit(X) before "
                "transform/inverse_transform")

    # -- core -----------------------------------------------------------
    @staticmethod
    def _symmetric_decorrelation(W):
        """W <- (W W^T)^(-1/2) W, via eigendecomposition of W W^T."""
        s, u = np.linalg.eigh(W @ W.T)
        s = np.clip(s, 1e-12, None)
        return (u * (1.0 / np.sqrt(s))) @ u.T @ W

    def fit(self, X):
        """Fit the unmixing matrix on X. Returns self."""
        X = self._validate_X(X)
        n_samples, n_features = X.shape
        n_components = (n_features if self.n_components is None
                        else self.n_components)
        if n_components > n_features:
            raise ValueError(
                f"n_components={n_components} cannot exceed "
                f"n_features={n_features}")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Whiten: eigendecomposition of the covariance matrix.
        cov = (Xc.T @ Xc) / n_samples
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-12, None)
        whitening = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
        self.whitening_ = whitening
        Xw = Xc @ whitening.T

        rng = np.random.default_rng(self.random_state)
        W = rng.standard_normal((n_components, n_features))
        W = self._symmetric_decorrelation(W)

        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            Y = Xw @ W.T
            g = np.tanh(Y)
            g_prime = 1.0 - g ** 2
            W_new = (g.T @ Xw) / n_samples - (
                g_prime.mean(axis=0)[:, None] * W)
            W_new = self._symmetric_decorrelation(W_new)
            # Convergence: rows of W_new align with rows of W (up to sign).
            delta = 1.0 - np.abs(np.diag(W_new @ W.T)).max()
            W = W_new
            self.n_iter_ = it
            if delta < self.tol:
                break

        self.components_ = W  # unmixing matrix in whitened space
        # Mixing matrix: X ≈ S @ mixing_.T + mean_, with S = Xw @ W.T.
        self.mixing_ = np.linalg.pinv(W @ whitening)
        return self

    def transform(self, X):
        """Project X onto the independent components. Returns (n_samples,
        n_components)."""
        self._check_fitted()
        X = self._validate_X(X)
        Xc = X - self.mean_
        Xw = Xc @ self.whitening_.T
        return Xw @ self.components_.T

    def fit_transform(self, X):
        """Fit on X and return the extracted components."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Reconstruct original-space data from component scores."""
        self._check_fitted()
        if isinstance(X, (list, tuple)):
            X = np.asarray(X, dtype=float)
        if not isinstance(X, np.ndarray) or X.dtype.kind not in "fiu":
            raise TypeError("X must be a numeric array")
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D, got {X.ndim}D with shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X must be non-empty, got shape {X.shape}")
        if X.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"Expected {self.components_.shape[0]} components, got "
                f"{X.shape[1]}")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or inf")
        return X @ self.mixing_.T + self.mean_
