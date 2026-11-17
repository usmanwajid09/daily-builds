"""Non-negative matrix factorization from scratch (Lee-Seung updates).

Factorize a non-negative matrix ``V`` (shape ``(n_samples, n_features)``)
into ``W`` (``(n_samples, n_components)``) and ``H`` (``(n_components,
n_features)``), both non-negative, such that ``V ≈ W @ H``.

Multiplicative updates (Lee & Seung, 2001), minimizing the squared
Frobenius error ``||V - W H||_F²``::

    H *= (W^T V) / (W^T W H + eps)
    W *= (V H^T) / (W H H^T + eps)

Because every update is a multiplication by a non-negative ratio, ``W``
and ``H`` stay non-negative as long as they start non-negative -- no
projection or clipping step is needed.
"""

import numpy as np

_EPS = 1e-10


class NMF:
    """Non-negative matrix factorization via multiplicative updates.

    Parameters
    ----------
    n_components : int
        Number of components (rank of the factorization). Must satisfy
        ``1 <= n_components <= min(V.shape)``.
    max_iter : int
        Maximum number of update sweeps.
    tol : float
        Convergence tolerance on the relative change of the
        reconstruction error between sweeps.
    random_state : int
        Seed for the random initialization of ``W`` and ``H``.

    Attributes
    ----------
    components_ : ndarray, shape (n_components, n_features)
        The fitted ``H`` matrix.
    W_ : ndarray, shape (n_samples, n_components)
        The fitted ``W`` matrix (also returned by ``fit_transform``).
    reconstruction_err_ : float
        Final Frobenius reconstruction error ``||V - W H||_F``.
    n_iter_ : int
        Number of update sweeps actually performed.
    """

    def __init__(self, n_components=2, max_iter=500, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # -- validation -----------------------------------------------------
    def _validate_V(self, V):
        V = np.asarray(V, dtype=float)
        if V.ndim != 2:
            raise ValueError(
                f"V must be 2D, got {V.ndim}D input with shape {V.shape}."
            )
        if V.size == 0 or V.shape[0] == 0 or V.shape[1] == 0:
            raise ValueError(f"V must be non-empty, got shape {V.shape}.")
        if not np.all(np.isfinite(V)):
            raise ValueError("V must contain only finite values (no NaN or inf).")
        if np.any(V < 0):
            raise ValueError("V must be non-negative; found negative entries.")
        return V

    def _validate_n_components(self, V):
        k = self.n_components
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise ValueError(f"n_components must be an integer, got {k!r}.")
        if k < 1:
            raise ValueError(f"n_components must be >= 1, got {k}.")
        if k > min(V.shape):
            raise ValueError(
                f"n_components ({k}) must be <= min(V.shape) = {min(V.shape)} "
                f"for V with shape {V.shape}."
            )

    def _check_fitted(self):
        if not hasattr(self, "components_"):
            raise RuntimeError("NMF is not fitted yet; call fit(V) first.")

    # -- fitting --------------------------------------------------------
    def fit(self, V):
        """Fit the factorization ``V ≈ W @ H``.

        Returns self.
        """
        V = self._validate_V(V)
        self._validate_n_components(V)
        m, n = V.shape
        k = self.n_components

        rng = np.random.default_rng(self.random_state)
        # Strictly positive start: a zero entry would stay zero forever
        # under multiplicative updates.
        W = rng.uniform(0.1, 1.0, size=(m, k))
        H = rng.uniform(0.1, 1.0, size=(k, n))

        prev_err = None
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            H *= (W.T @ V) / (W.T @ W @ H + _EPS)
            W *= (V @ H.T) / (W @ H @ H.T + _EPS)
            err = float(np.linalg.norm(V - W @ H))
            self.n_iter_ = it
            if prev_err is not None and prev_err > 0:
                if abs(prev_err - err) / prev_err < self.tol:
                    break
            prev_err = err

        self.W_ = W
        self.components_ = H
        self.reconstruction_err_ = float(np.linalg.norm(V - W @ H))
        return self

    def fit_transform(self, V):
        """Fit the model and return ``W`` (shape ``(n_samples, n_components)``)."""
        self.fit(V)
        return self.W_

    def transform(self, V):
        """Encode new data with the fitted components.

        Returns ``W_new`` (shape ``(n_samples, n_components)``) such that
        ``V ≈ W_new @ components_``. Only the coefficients are updated --
        ``components_`` (``H``) stays fixed -- using the same multiplicative
        update with ``H`` held constant, run for up to ``max_iter`` sweeps
        (or until the same tolerance is met). This is the standard
        sklearn-style ``transform``; equivalently, in the transposed layout
        it is "fit H with W fixed".
        """
        self._check_fitted()
        V = self._validate_V(V)
        if V.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"V has {V.shape[1]} features, but the model was fitted with "
                f"{self.components_.shape[1]} features."
            )
        H = self.components_
        rng = np.random.default_rng(self.random_state)
        W = rng.uniform(0.1, 1.0, size=(V.shape[0], self.n_components))
        prev_err = None
        for _ in range(self.max_iter):
            W *= (V @ H.T) / (W @ H @ H.T + _EPS)
            err = float(np.linalg.norm(V - W @ H))
            if prev_err is not None and prev_err > 0:
                if abs(prev_err - err) / prev_err < self.tol:
                    break
            prev_err = err
        return W

    def inverse_transform(self, W):
        """Reconstruct data from coefficients: ``W @ components_``."""
        self._check_fitted()
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[1] != self.n_components:
            raise ValueError(
                f"W must have shape (n_samples, {self.n_components}), "
                f"got {W.shape}."
            )
        return W @ self.components_

    def score(self, V):
        """Explained variance (R²-like) of the reconstruction.

        Returns ``1 - ||V - W H||_F² / ||V||_F²``: 1.0 is a perfect
        reconstruction, 0.0 means no better than reconstructing with
        zeros. (This is explained variance, not the negative error.)
        """
        self._check_fitted()
        V = self._validate_V(V)
        recon = self.inverse_transform(self.transform(V))
        ss_res = float(np.sum((V - recon) ** 2))
        ss_tot = float(np.sum(V ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
