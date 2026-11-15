"""Linear Discriminant Analysis (LDA) classifier, from scratch with numpy only.

LDA models each class as a Gaussian with its own mean but a *shared*
covariance matrix, plus a class prior. A new point is scored by the
log posterior of each class,
``log prior_k - 0.5 * (x - mu_k)^T Sigma^-1 (x - mu_k)``,
and assigned to the highest-scoring class. Because the covariance is
shared, the quadratic terms cancel and the decision boundary between
any two classes is a hyperplane, which makes LDA fast, stable, and
surprisingly strong when classes are roughly Gaussian with similar
spread. The shared covariance is estimated from within-class scatter
and inverted with a pseudo-inverse, so it copes with a singular
covariance; an optional shrinkage parameter blends it toward a scaled
identity for extra stability on small samples.
"""
from __future__ import annotations

import numpy as np


class LDAClassifier:
    """Linear Discriminant Analysis classifier.

    Parameters
    ----------
    shrinkage : float
        Amount of shrinkage applied to the shared covariance matrix
        toward a scaled identity: ``(1 - shrinkage) * Sigma +
        shrinkage * (trace(Sigma) / n_features) * I``. Must lie in
        [0, 1]; 0 means no shrinkage. Useful when samples are few
        relative to features.
    priors : array-like of float, optional
        Class prior probabilities, in the order of the sorted unique
        class labels. If omitted, priors are estimated from the class
        frequencies in the training data.
    """

    def __init__(self, shrinkage: float = 0.0, priors=None):
        if not isinstance(shrinkage, (int, float)) or isinstance(shrinkage, bool) \
                or not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be a number in [0, 1]")
        if priors is not None:
            priors = np.asarray(priors, dtype=float)
            if priors.ndim != 1 or priors.size == 0:
                raise ValueError("priors must be a non-empty 1D array")
            if np.any(priors <= 0) or not np.isclose(priors.sum(), 1.0):
                raise ValueError("priors must be positive and sum to 1")
        self.shrinkage = float(shrinkage)
        self.priors = priors
        self.classes_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LDAClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("LDA needs at least 2 classes")
        if self.priors is not None and self.priors.size != classes.size:
            raise ValueError(
                f"priors has {self.priors.size} entries but there are "
                f"{classes.size} classes"
            )

        n, p = x.shape
        means = np.zeros((classes.size, p))
        # Pooled within-class covariance.
        cov = np.zeros((p, p))
        for k, c in enumerate(classes):
            xk = x[y == c]
            if xk.shape[0] < 2:
                raise ValueError(
                    f"class {c!r} has only {xk.shape[0]} sample(s); "
                    "each class needs at least 2"
                )
            means[k] = xk.mean(axis=0)
            centred = xk - means[k]
            cov += centred.T @ centred
        cov /= (n - classes.size)

        if self.shrinkage > 0.0:
            target = np.trace(cov) / p * np.eye(p)
            cov = (1.0 - self.shrinkage) * cov + self.shrinkage * target

        if self.priors is not None:
            priors = self.priors.copy()
        else:
            priors = np.array([np.mean(y == c) for c in classes])

        self.classes_ = classes
        self.means_ = means
        self.covariance_ = cov
        self.priors_ = priors
        self._cov_inv = np.linalg.pinv(cov)
        return self

    def _log_posterior(self, x: np.ndarray) -> np.ndarray:
        # Mahalanobis distance to each class mean, plus log prior.
        diff = x[:, None, :] - self.means_[None, :, :]  # (n, K, p)
        maha = np.einsum("nkp,pq,nkq->nk", diff, self._cov_inv, diff)
        return np.log(self.priors_) - 0.5 * maha

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("model is not fitted yet; call fit() first")
        x = self._check_predict_input(x)
        return self.classes_[np.argmax(self._log_posterior(x), axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities (rows sum to 1)."""
        if self.classes_ is None:
            raise ValueError("model is not fitted yet; call fit() first")
        x = self._check_predict_input(x)
        log_post = self._log_posterior(x)
        log_post -= log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        return probs / probs.sum(axis=1, keepdims=True)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(x) == np.asarray(y)))

    def _check_predict_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[1] != self.means_.shape[1]:
            raise ValueError(
                f"x has {x.shape[1]} features but the model was fitted on "
                f"{self.means_.shape[1]}"
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        return x
