"""Factor analysis via the EM algorithm (numpy + stdlib only).

Model: x = W @ z + mu + eps, with z ~ N(0, I) and eps ~ N(0, Psi),
Psi diagonal. Fitted by expectation-maximisation.
"""

import numpy as np

__all__ = ["FactorAnalysis"]


class FactorAnalysis:
    """Factor analysis fitted with the EM algorithm.

    Parameters
    ----------
    n_factors : int, >= 1
        Number of latent factors (must be < n_features at fit time).
    max_iter : int, default 100
        Maximum number of EM iterations.
    tol : float, default 1e-4
        Stop when the log-likelihood improves by less than ``tol``.
    random_state : int or None
        Seed for the small random jitter added to the PCA-based
        initialisation, so fits are reproducible.
    """

    def __init__(self, n_factors, max_iter=100, tol=1e-4, random_state=None):
        if not isinstance(n_factors, (int, np.integer)) or int(n_factors) < 1:
            raise ValueError("n_factors must be an integer >= 1")
        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter must be an integer >= 1")
        if tol is None or float(tol) <= 0:
            raise ValueError("tol must be positive")
        self.n_factors = int(n_factors)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.components_ = None      # loadings W, shape (n_features, n_factors)
        self.noise_variance_ = None  # Psi diagonal, shape (n_features,)
        self.mean_ = None            # mu, shape (n_features,)
        self.n_iter_ = 0
        self.log_likelihood_history_ = []

    # -- validation -----------------------------------------------------
    def _check_fitted(self):
        if self.components_ is None:
            raise ValueError("FactorAnalysis is not fitted yet; call fit(X) first")

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be non-empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values (no NaN or inf)")
        return X

    def _check_X_features(self, X):
        X = self._validate_X(X)
        if X.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted on "
                f"{self.components_.shape[0]}"
            )
        return X

    # -- log-likelihood -------------------------------------------------
    def _log_likelihood(self, X, W, psi, mu):
        """Total log-likelihood of X under N(mu, W W^T + diag(psi))."""
        d = X.shape[1]
        cov = W @ W.T + np.diag(psi)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
        Xc = X - mu
        try:
            solved = np.linalg.solve(cov, Xc.T)  # (d, n)
        except np.linalg.LinAlgError:
            return -np.inf
        quad = np.einsum("ij,ij->i", Xc, solved.T)
        return float(
            -0.5 * X.shape[0] * (d * np.log(2.0 * np.pi) + logdet) - 0.5 * quad.sum()
        )

    def log_likelihood(self, X):
        """Total log-likelihood of X under the fitted model."""
        self._check_fitted()
        X = self._check_X_features(X)
        return self._log_likelihood(X, self.components_, self.noise_variance_, self.mean_)

    def score(self, X):
        """Mean per-sample log-likelihood of X under the fitted model."""
        X = self._validate_X(X)
        return self.log_likelihood(X) / X.shape[0]

    # -- fitting --------------------------------------------------------
    def fit(self, X):
        """Fit the factor model to X with EM. Returns self."""
        X = self._validate_X(X)
        n, d = X.shape
        k = self.n_factors
        if k >= d:
            raise ValueError(
                f"n_factors ({k}) must be < n_features ({d})"
            )
        if n <= k:
            raise ValueError(
                f"n_samples ({n}) must be > n_factors ({k})"
            )

        rng = np.random.default_rng(self.random_state)
        mu = X.mean(axis=0)
        Xc = X - mu
        # Sample covariance (ML, divisor n).
        S = (Xc.T @ Xc) / n

        # PCA-based initialisation: top-k eigendirections, scaled so that
        # W W^T explains the top-k variance; Psi takes the residual.
        eigvals, eigvecs = np.linalg.eigh(S)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        resid = eigvals[k:].mean() if k < d else 0.0
        W = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k] - resid, 0.1))
        W = W + rng.normal(scale=1e-3, size=W.shape)
        psi = np.clip(np.diag(S) - np.sum(W ** 2, axis=1), 1e-6, None)

        prev_ll = -np.inf
        self.log_likelihood_history_ = []
        for it in range(1, self.max_iter + 1):
            # E-step: posterior z | x ~ N(m, G), G = (I + W^T Psi^-1 W)^-1.
            psi_inv = 1.0 / psi
            Wt_psi_inv = W.T * psi_inv  # (k, d)
            G = np.linalg.inv(np.eye(k) + Wt_psi_inv @ W)
            Ez = (Xc * psi_inv) @ W @ G.T  # (n, k) posterior means
            # sum_n E[z z^T] = n*G + Ez^T Ez
            Ezz_sum = n * G + Ez.T @ Ez
            # M-step: W = (Xc^T Ez) (Ezz_sum)^-1
            XEz = Xc.T @ Ez  # (d, k)
            W_new = XEz @ np.linalg.inv(Ezz_sum)
            # Psi = diag(S - W_new @ XEz^T / n), floored at 1e-6.
            psi_new = np.diag(S - (W_new @ XEz.T) / n)
            psi_new = np.clip(psi_new, 1e-6, None)
            W, psi = W_new, psi_new

            ll = self._log_likelihood(X, W, psi, mu)
            self.log_likelihood_history_.append(ll)
            self.n_iter_ = it
            if ll - prev_ll < self.tol and it > 1:
                break
            prev_ll = ll

        self.components_ = W
        self.noise_variance_ = psi
        self.mean_ = mu
        return self

    # -- inference ------------------------------------------------------
    def transform(self, X):
        """Posterior mean of the latent factors for each row of X."""
        self._check_fitted()
        X = self._check_X_features(X)
        W, psi, mu = self.components_, self.noise_variance_, self.mean_
        psi_inv = 1.0 / psi
        G = np.linalg.inv(np.eye(self.n_factors) + (W.T * psi_inv) @ W)
        return ((X - mu) * psi_inv) @ W @ G.T

    def reconstruct(self, X):
        """Reconstruct X from its posterior factor means: W @ z + mu."""
        Z = self.transform(X)
        return Z @ self.components_.T + self.mean_

    def fit_transform(self, X):
        """Fit on X and return the posterior factor means."""
        return self.fit(X).transform(X)
