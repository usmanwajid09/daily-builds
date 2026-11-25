"""t-SNE (t-distributed Stochastic Neighbour Embedding), from scratch.

Reduces high-dimensional data to a low-dimensional embedding by matching
pairwise similarities: Gaussian affinities in the input space, Student-t
affinities in the embedding space, minimising the KL divergence between
them by gradient descent.
"""
from __future__ import annotations

import numpy as np


class TSNE:
    """t-SNE dimensionality reduction, implemented with numpy only.

    Parameters
    ----------
    n_components : int
        Dimension of the embedding (usually 2).
    perplexity : float
        Effective number of neighbours each point attends to. Must satisfy
        ``1 <= perplexity < n_samples``.
    n_iter : int
        Number of gradient-descent iterations.
    learning_rate : float
        Step size for the gradient updates.
    random_state : int or None
        Seed for the initial embedding. ``None`` gives a random start.
    early_exaggeration : float
        Factor by which the input affinities are multiplied during the
        first ``exaggeration_iters`` iterations, to force clusters apart.
    exaggeration_iters : int
        Number of iterations for which early exaggeration applies.
    momentum : float
        Momentum used during the early-exaggeration phase.
    final_momentum : float
        Momentum used after the early-exaggeration phase.
    """

    def __init__(self, n_components=2, perplexity=30.0, n_iter=1000,
                 learning_rate=200.0, random_state=None,
                 early_exaggeration=12.0, exaggeration_iters=250,
                 momentum=0.8, final_momentum=0.8):
        if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
            raise TypeError(f"n_components must be a positive int, got {n_components!r}")
        if n_components < 1:
            raise ValueError(f"n_components must be a positive int, got {n_components!r}")
        if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, bool) or n_iter < 1:
            raise ValueError(f"n_iter must be a positive int, got {n_iter!r}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}")
        if early_exaggeration < 1:
            raise ValueError(f"early_exaggeration must be >= 1, got {early_exaggeration!r}")
        if not isinstance(exaggeration_iters, (int, np.integer)) or exaggeration_iters < 0:
            raise ValueError(f"exaggeration_iters must be a non-negative int, got {exaggeration_iters!r}")
        if not 0.0 <= momentum < 1.0 or not 0.0 <= final_momentum < 1.0:
            raise ValueError("momentum and final_momentum must lie in [0, 1)")
        self.n_components = int(n_components)
        self.perplexity = float(perplexity)
        self.n_iter = int(n_iter)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.early_exaggeration = float(early_exaggeration)
        self.exaggeration_iters = int(exaggeration_iters)
        self.momentum = float(momentum)
        self.final_momentum = float(final_momentum)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_X(X):
        if isinstance(X, (list, tuple)):
            try:
                X = np.asarray(X, dtype=float)
            except (TypeError, ValueError):
                raise TypeError("X must be a numeric 2D array")
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X).__name__}")
        if X.dtype == object or not np.issubdtype(X.dtype, np.number):
            raise TypeError("X must be numeric")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D with shape {X.shape}")
        if X.size == 0 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be non-empty")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        return X.astype(float, copy=False)

    def _validate_perplexity(self, n_samples):
        if self.perplexity < 1:
            raise ValueError(
                f"perplexity must be >= 1, got {self.perplexity}")
        if self.perplexity >= n_samples:
            raise ValueError(
                f"perplexity ({self.perplexity}) must be < n_samples ({n_samples})")

    # ------------------------------------------------------------------
    # affinities
    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_sq_dists(X):
        sq = np.sum(X * X, axis=1, keepdims=True)
        D = sq + sq.T - 2.0 * (X @ X.T)
        return np.maximum(D, 0.0)

    def _conditional_affinities(self, D):
        """Gaussian conditional affinities p_{j|i} via per-point sigma search."""
        n = D.shape[0]
        log_perp = np.log(self.perplexity)
        P = np.zeros((n, n))
        for i in range(n):
            # Binary search on beta = 1 / (2 sigma^2) so that the Shannon
            # entropy of the conditional distribution equals log(perplexity).
            beta_min, beta_max = -np.inf, np.inf
            beta = 1.0
            Di = D[i].copy()
            Di[i] = np.inf  # exclude self
            for _ in range(50):
                p = np.exp(-Di * beta)
                p[i] = 0.0
                sum_p = p.sum()
                if sum_p == 0.0:  # all distances huge: fall back to uniform
                    p = np.ones(n)
                    p[i] = 0.0
                    p /= p.sum()
                    break
                p /= sum_p
                entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
                diff = entropy - log_perp
                if abs(diff) < 1e-5:
                    break
                if diff > 0:  # entropy too high -> sharpen (increase beta)
                    beta_min = beta
                    beta = beta * 2.0 if np.isinf(beta_max) else (beta + beta_max) / 2.0
                else:  # entropy too low -> widen (decrease beta)
                    beta_max = beta
                    beta = beta / 2.0 if np.isinf(beta_min) else (beta + beta_min) / 2.0
            P[i] = p
        return P

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, X):
        X = self._validate_X(X)
        n = X.shape[0]
        self._validate_perplexity(n)

        D = self._pairwise_sq_dists(X)
        P = self._conditional_affinities(D)
        # Symmetrise and normalise; floor so no affinity is exactly zero.
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)
        P /= P.sum()

        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(0.0, 1e-4, size=(n, self.n_components))
        Y_update = np.zeros_like(Y)

        for it in range(self.n_iter):
            # Student-t affinities in the embedding space.
            D_y = self._pairwise_sq_dists(Y)
            num = 1.0 / (1.0 + D_y)
            np.fill_diagonal(num, 0.0)
            Q = num / num.sum()
            Q = np.maximum(Q, 1e-12)

            P_eff = P * (self.early_exaggeration if it < self.exaggeration_iters else 1.0)
            # Gradient of the KL divergence: 4 * sum_j (p_ij - q_ij) * q_ij * Z * (y_i - y_j)
            # with Z the normaliser of num. (p - q) * num gives the same up to Z.
            PQ = (P_eff - Q) * num
            grad = 4.0 * (np.diag(PQ.sum(axis=1)) - PQ) @ Y

            # Clip each point's gradient to norm 4 for stability.
            norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = grad * np.minimum(1.0, 4.0 / np.maximum(norms, 1e-12))

            momentum = self.momentum if it < self.exaggeration_iters else self.final_momentum
            Y_update = momentum * Y_update - self.learning_rate * grad
            Y = Y + Y_update
            # Recentre to keep the embedding from drifting.
            Y -= Y.mean(axis=0, keepdims=True)

        self.embedding_ = Y
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
