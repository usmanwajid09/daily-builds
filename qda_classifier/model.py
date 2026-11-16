"""Quadratic Discriminant Analysis (QDA) classifier, from scratch with numpy only.

QDA models each class as a Gaussian with its own mean *and* its own
covariance matrix, plus a class prior. A new point is scored by the log
posterior of each class,

``log prior_k - 0.5 * log|Sigma_k| - 0.5 * (x - mu_k)^T Sigma_k^-1 (x - mu_k)``,

and assigned to the highest-scoring class. Because each class has its own
covariance, the quadratic terms do not cancel, so the decision boundary
between two classes is a conic section (ellipse, parabola, or hyperbola)
instead of a hyperplane. That makes QDA strictly more flexible than LDA:
it can separate classes that share a mean but differ in spread (a tight
cluster sitting inside a wide one), which no linear boundary can do.

The price is more parameters to estimate (one covariance per class), so
QDA needs more data per class than LDA and can overfit on small samples.
A small ridge term ``reg`` is added to each covariance diagonal for
numerical stability; singular covariances fall back to a pseudo-inverse.
"""

from __future__ import annotations

import numpy as np


class QDAClassifier:
    """Quadratic Discriminant Analysis classifier.

    Parameters
    ----------
    reg : float, default=1e-6
        Ridge regularisation added to the diagonal of each class
        covariance matrix. Must be non-negative; 0 means no
        regularisation. A small value keeps nearly-singular
        covariances invertible.
    priors : array-like of float, optional
        Class prior probabilities, in the order of the sorted unique
        class labels. If omitted, priors are estimated from the class
        frequencies in the training data.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique class labels seen in ``fit``.
    means_ : ndarray of shape (n_classes, n_features)
        Per-class mean vectors.
    covariances_ : ndarray of shape (n_classes, n_features, n_features)
        Per-class covariance matrices (with ``reg`` added to the diagonal).
    priors_ : ndarray of shape (n_classes,)
        Per-class prior probabilities.
    """

    def __init__(self, reg: float = 1e-6, priors=None):
        if isinstance(reg, bool) or not isinstance(reg, (int, float)):
            raise ValueError("reg must be a number")
        if reg < 0:
            raise ValueError("reg must be non-negative")
        if priors is not None:
            priors = np.asarray(priors, dtype=float)
            if priors.ndim != 1 or priors.size == 0:
                raise ValueError("priors must be a non-empty 1D array")
            if np.any(priors <= 0):
                raise ValueError("priors must all be positive")
            if not np.isclose(priors.sum(), 1.0):
                raise ValueError("priors must sum to 1")
        self.reg = float(reg)
        self.priors = priors
        self.classes_ = None
        self.means_ = None
        self.covariances_ = None
        self.priors_ = None

    # -- fitting ---------------------------------------------------------
    def fit(self, X, y):
        """Fit the QDA model: per-class means, covariances and priors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("X and y must be non-empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: {X.shape[0]} != {y.shape[0]}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("need at least 2 classes to fit a classifier")

        if self.priors is not None and self.priors.size != classes.size:
            raise ValueError(
                f"priors has length {self.priors.size} but there are "
                f"{classes.size} classes"
            )

        n_features = X.shape[1]
        means = np.empty((classes.size, n_features))
        covariances = np.empty((classes.size, n_features, n_features))
        counts = np.empty(classes.size, dtype=float)

        for k, c in enumerate(classes):
            Xk = X[y == c]
            counts[k] = Xk.shape[0]
            means[k] = Xk.mean(axis=0)
            if Xk.shape[0] == 1:
                # A single sample has no spread; fall back to the ridge.
                cov = np.zeros((n_features, n_features))
            else:
                cov = np.atleast_2d(np.cov(Xk, rowvar=False))
            cov = cov + self.reg * np.eye(n_features)
            covariances[k] = cov

        if self.priors is not None:
            priors = np.asarray(self.priors, dtype=float)
        else:
            priors = counts / counts.sum()

        self.classes_ = classes
        self.means_ = means
        self.covariances_ = covariances
        self.priors_ = priors
        return self

    # -- helpers ---------------------------------------------------------
    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("QDAClassifier is not fitted yet; call fit() first")

    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must be non-empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted on "
                f"{self.means_.shape[1]}"
            )
        return X

    def _discriminants(self, X):
        """Log-posterior score of each class for every row of X.

        Uses ``slogdet`` for the log-determinant and ``solve`` (not an
        explicit inverse) for the Mahalanobis term; a singular or
        non-positive-definite covariance falls back to a pseudo-inverse
        and a log-determinant from the positive eigenvalues.
        """
        n = X.shape[0]
        scores = np.empty((n, self.classes_.size))
        for k in range(self.classes_.size):
            diff = X - self.means_[k]
            Sigma = self.covariances_[k]
            sign, logdet = np.linalg.slogdet(Sigma)
            try:
                if sign <= 0:
                    raise np.linalg.LinAlgError(
                        "covariance is not positive definite"
                    )
                # solve(Sigma, diff.T) is Sigma^-1 @ diff.T, without forming
                # the inverse explicitly.
                solved = np.linalg.solve(Sigma, diff.T).T
                mahal = np.einsum("ij,ij->i", diff, solved)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(Sigma)
                mahal = np.einsum("ij,jk,ik->i", diff, inv, diff)
                eig = np.linalg.eigvalsh(Sigma)
                eig = eig[eig > 0]
                logdet = float(np.sum(np.log(eig))) if eig.size else 0.0
            scores[:, k] = np.log(self.priors_[k]) - 0.5 * logdet - 0.5 * mahal
        return scores

    # -- prediction ------------------------------------------------------
    def predict(self, X):
        """Predict the class label for each row of X."""
        self._check_fitted()
        X = self._check_X(X)
        scores = self._discriminants(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities for each row of X.

        Softmax over the log-posterior discriminants, with the row max
        subtracted first for numerical stability. Rows sum to 1.
        """
        self._check_fitted()
        X = self._check_X(X)
        scores = self._discriminants(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
