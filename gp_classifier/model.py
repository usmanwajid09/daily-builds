"""Gaussian process classifier with a Laplace approximation.

Binary classifier with an RBF (squared-exponential) kernel and a
Bernoulli / sigmoid likelihood. The posterior over the latent function
values is approximated by a Gaussian centred at the MAP estimate, found
by Newton iterations (Rasmussen & Williams, Algorithm 3.1). Predictive
probabilities use the MacKay probit approximation, which folds the
latent predictive variance into the sigmoid.

NumPy + standard library only.
"""

import numpy as np


def _sigmoid(z):
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _cholesky_with_jitter(matrix, jitters=(1e-10, 1e-8, 1e-6, 1e-4, 1e-2)):
    """Cholesky factorisation, adding jitter until it succeeds."""
    n = matrix.shape[0]
    for jitter in jitters:
        try:
            return np.linalg.cholesky(matrix + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky failed even with jitter")


class GaussianProcessClassifier:
    """Gaussian process classifier (Laplace approximation, RBF kernel).

    Parameters
    ----------
    length_scale : float > 0
        RBF kernel length scale.
    signal_variance : float > 0
        RBF kernel signal variance (k(x, x) for any x).
    max_iter : int, default 50
        Maximum Newton iterations for the Laplace approximation.
    tol : float, default 1e-5
        Convergence tolerance on the largest change in the latent
        function values between Newton iterations.
    """

    def __init__(self, length_scale=1.0, signal_variance=1.0,
                 max_iter=50, tol=1e-5):
        if not np.isfinite(length_scale) or length_scale <= 0:
            raise ValueError("length_scale must be a finite float > 0")
        if not np.isfinite(signal_variance) or signal_variance <= 0:
            raise ValueError("signal_variance must be a finite float > 0")
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a finite float > 0")
        self.length_scale = float(length_scale)
        self.signal_variance = float(signal_variance)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    # -- kernel ---------------------------------------------------------
    def _kernel(self, X1, X2):
        """RBF kernel matrix between rows of X1 and rows of X2."""
        sq1 = np.sum(X1 ** 2, axis=1)[:, None]
        sq2 = np.sum(X2 ** 2, axis=1)[None, :]
        sq_dist = np.maximum(sq1 + sq2 - 2.0 * X1 @ X2.T, 0.0)
        return self.signal_variance * np.exp(
            -sq_dist / (2.0 * self.length_scale ** 2))

    # -- validation -----------------------------------------------------
    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _validate_fit_inputs(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: {X.shape[0]} != {len(y)}")
        if X.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary labels 0 and 1")
        return X, y.astype(float)

    def _check_fitted(self):
        if not hasattr(self, "X_train_"):
            raise ValueError("GaussianProcessClassifier is not fitted yet; "
                             "call fit(X, y) first")

    def _validate_predict_X(self, X):
        self._check_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected "
                f"{self.X_train_.shape[1]}")
        return X

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        """Fit by Laplace approximation (Newton iterations on latent f)."""
        X, y = self._validate_fit_inputs(X, y)
        n = X.shape[0]
        K = self._kernel(X, X)

        # Newton iterations (Rasmussen & Williams, Algorithm 3.1).
        f = np.zeros(n)
        alpha = np.zeros(n)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            pi = _sigmoid(f)
            W = np.clip(pi * (1.0 - pi), 1e-12, None)
            sqrt_w = np.sqrt(W)
            # B = I + W^{1/2} K W^{1/2}
            B = np.eye(n) + (K * sqrt_w[:, None]) * sqrt_w[None, :]
            L = _cholesky_with_jitter(B)
            b = W * f + (y - pi)
            # a = b - W^{1/2} L^{-T} L^{-1} W^{1/2} K b ;  f_new = K a
            Kb = K @ b
            tmp = np.linalg.solve(L, sqrt_w * Kb)
            alpha = b - sqrt_w * np.linalg.solve(L.T, tmp)
            f_new = K @ alpha
            if np.max(np.abs(f_new - f)) < self.tol:
                f = f_new
                break
            f = f_new

        # Recompute the final Cholesky factor at the MAP estimate, so
        # predict() uses the B belonging to the stored f / W.
        pi = _sigmoid(f)
        W = np.clip(pi * (1.0 - pi), 1e-12, None)
        sqrt_w = np.sqrt(W)
        B = np.eye(n) + (K * sqrt_w[:, None]) * sqrt_w[None, :]
        self.L_ = _cholesky_with_jitter(B)
        self.sqrt_w_ = sqrt_w
        self.alpha_ = alpha
        self.f_ = f
        self.X_train_ = X
        self.y_train_ = y.astype(int)
        self.n_iter_ = n_iter
        return self

    # -- prediction -----------------------------------------------------
    def _predict_latent(self, X):
        """Predictive latent mean and variance for rows of X."""
        X = self._validate_predict_X(X)
        K_star = self._kernel(self.X_train_, X)          # (n_train, n_test)
        mean = K_star.T @ self.alpha_
        # V = L^{-1} W^{1/2} K_* ; var = k_** - sum(V^2, axis=0)
        V = np.linalg.solve(self.L_, self.sqrt_w_[:, None] * K_star)
        var = self.signal_variance - np.sum(V ** 2, axis=0)
        var = np.clip(var, 0.0, None)
        return mean, var

    def predict_proba(self, X):
        """Class probabilities, columns [P(y=0), P(y=1)].

        Uses the MacKay probit approximation:
        pi ~= sigmoid(mean / sqrt(1 + pi/8 * var)).
        """
        mean, var = self._predict_latent(X)
        pi = _sigmoid(mean / np.sqrt(1.0 + (np.pi / 8.0) * var))
        pi = np.clip(pi, 0.0, 1.0)
        return np.column_stack([1.0 - pi, pi])

    def predict(self, X):
        """Predicted class labels (0 or 1)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        """Mean accuracy on (X, y)."""
        X = self._validate_predict_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be a 1D array matching X in length")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary labels 0 and 1")
        return float(np.mean(self.predict(X) == y))
