"""Gaussian mixture model, from scratch with numpy only.

Fits a mixture of multivariate Gaussians with full covariance matrices by
expectation-maximisation: the E-step computes soft responsibilities for
each point under each component, and the M-step re-estimates the weights,
means and covariances from those responsibilities. A small
``reg_covar`` is added to every covariance diagonal so the likelihood
stays finite even if a component collapses onto a few points.
"""
from __future__ import annotations

import numpy as np


class GaussianMixture:
    """Gaussian mixture model with full covariances, fit by EM.

    Parameters
    ----------
    n_components : int
        Number of mixture components. Must be at least 1 and no more than
        the number of samples.
    max_iter : int
        Maximum number of EM iterations. Must be at least 1.
    tol : float
        Convergence tolerance on the log-likelihood improvement.
        Must be positive.
    reg_covar : float
        Non-negative regularisation added to the diagonal of each
        covariance matrix. Must be positive.
    random_state : int or None
        Seed for the random initialisation of the component means.
    """

    def __init__(self, n_components: int = 2, max_iter: int = 100,
                 tol: float = 1e-3, reg_covar: float = 1e-6,
                 random_state: int | None = None):
        if not isinstance(n_components, int) or isinstance(n_components, bool) \
                or n_components < 1:
            raise ValueError("n_components must be an integer of at least 1")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) \
                or max_iter < 1:
            raise ValueError("max_iter must be an integer of at least 1")
        if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol <= 0:
            raise ValueError("tol must be a positive number")
        if (not isinstance(reg_covar, (int, float)) or isinstance(reg_covar, bool)
                or reg_covar <= 0):
            raise ValueError("reg_covar must be a positive number")
        if random_state is not None and (
                not isinstance(random_state, int) or isinstance(random_state, bool)):
            raise ValueError("random_state must be an integer or None")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.random_state = random_state
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
        self.n_iter_: int | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray) -> "GaussianMixture":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        n_samples, n_features = x.shape
        if self.n_components > n_samples:
            raise ValueError(
                f"n_components={self.n_components} exceeds n_samples={n_samples}"
            )
        rng = np.random.default_rng(self.random_state)
        # k-means++ style initialisation: the first mean is a random point
        # and each next mean is drawn with probability proportional to its
        # squared distance from the nearest mean chosen so far. This keeps
        # the components apart; starting them at nearby random points lets
        # EM collapse them all onto the global mean.
        means = np.empty((self.n_components, n_features))
        means[0] = x[rng.integers(n_samples)]
        closest_sq = np.sum((x - means[0]) ** 2, axis=1)
        for k in range(1, self.n_components):
            total = closest_sq.sum()
            if total == 0.0:  # all points identical: fall back to uniform
                idx = rng.integers(n_samples)
            else:
                idx = rng.choice(n_samples, p=closest_sq / total)
            means[k] = x[idx]
            closest_sq = np.minimum(
                closest_sq, np.sum((x - means[k]) ** 2, axis=1)
            )
        overall_cov = np.cov(x, rowvar=False)
        if n_features == 1:
            overall_cov = np.atleast_2d(overall_cov)
        covariances = np.array([
            overall_cov + self.reg_covar * np.eye(n_features)
            for _ in range(self.n_components)
        ])
        weights = np.full(self.n_components, 1.0 / self.n_components)
        prev_ll = None
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            log_resp, ll = self._e_step(x, weights, means, covariances)
            resp = np.exp(log_resp)
            self._m_step(x, resp, means, covariances, weights)
            self.n_iter_ = it
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        _, self.log_likelihood_ = self._e_step(x, weights, means, covariances)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Posterior probability of each component for each sample."""
        self._check_fitted()
        x = self._check_x(x)
        log_resp, _ll = self._e_step(x, self.weights_, self.means_,
                                     self.covariances_)
        return np.exp(log_resp)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most likely component index for each sample."""
        return np.argmax(self.predict_proba(x), axis=1)

    def score(self, x: np.ndarray) -> float:
        """Average log-likelihood of the data under the fitted model."""
        self._check_fitted()
        x = self._check_x(x)
        _log_resp, ll = self._e_step(x, self.weights_, self.means_,
                                     self.covariances_)
        return ll / x.shape[0]

    # -- internals ------------------------------------------------------
    def _e_step(self, x, weights, means, covariances):
        n_samples = x.shape[0]
        log_prob = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = (
                np.log(weights[k])
                + self._log_gaussian(x, means[k], covariances[k])
            )
        log_prob_norm = self._logsumexp(log_prob, axis=1)
        log_resp = log_prob - log_prob_norm[:, np.newaxis]
        return log_resp, float(np.sum(log_prob_norm))

    def _m_step(self, x, resp, means, covariances, weights):
        n_samples, n_features = x.shape
        nk = resp.sum(axis=0) + 1e-10
        weights[:] = nk / n_samples
        for k in range(self.n_components):
            means[k] = (resp[:, k, np.newaxis] * x).sum(axis=0) / nk[k]
            diff = x - means[k]
            covariances[k] = (
                (resp[:, k, np.newaxis] * diff).T @ diff / nk[k]
                + self.reg_covar * np.eye(n_features)
            )

    def _log_gaussian(self, x, mean, cov):
        n_features = x.shape[1]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            # Should not happen with reg_covar, but stay safe.
            cov = cov + self.reg_covar * np.eye(n_features)
            sign, logdet = np.linalg.slogdet(cov)
        diff = x - mean
        try:
            solved = np.linalg.solve(cov, diff.T).T
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(cov) @ diff.T
            solved = solved.T
        maha = np.sum(diff * solved, axis=1)
        return -0.5 * (n_features * np.log(2 * np.pi) + logdet + maha)

    @staticmethod
    def _logsumexp(a, axis):
        m = np.max(a, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis=axis)

    def _check_fitted(self):
        if self.weights_ is None or self.means_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.means_.shape[1]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.means_.shape[1])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x
