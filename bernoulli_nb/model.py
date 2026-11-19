"""Bernoulli naive Bayes classifier, from scratch with numpy only.

Bernoulli naive Bayes models binary features: each feature is treated
as a coin flip whose probability of coming up 1 depends on the class.
It is the classic model for text classification with binary
word-presence features ("does this word appear in the document?").

With Laplace smoothing (``alpha``), the probability of feature ``j``
being 1 in class ``c`` is

    P(x_j = 1 | c) = (count of class-c samples with x_j = 1 + alpha)
                     / (count of class-c samples + 2 * alpha)

and the class prediction is the argmax of the log prior plus the sum
of the log feature likelihoods.
"""

from __future__ import annotations

import numpy as np


class BernoulliNB:
    """Bernoulli naive Bayes classifier for binary features.

    Parameters
    ----------
    alpha : float
        Laplace smoothing parameter. Must be positive. Default 1.0.
    fit_prior : bool
        If True, learn the class priors from the data; otherwise use a
        uniform prior. Default True.
    """

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.classes_ = None
        self.class_count_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_count_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if not np.isin(X, [0, 1]).all():
            raise ValueError("BernoulliNB requires binary features (0 or 1 only)")
        self.classes_, class_count = np.unique(y, return_counts=True)
        if len(self.classes_) < 2:
            raise ValueError("Need at least 2 classes to fit a classifier")
        self.class_count_ = class_count.astype(float)
        n_features = X.shape[1]
        self.feature_count_ = np.zeros((len(self.classes_), n_features))
        for i, c in enumerate(self.classes_):
            self.feature_count_[i] = X[y == c].sum(axis=0)
        # P(x_j = 1 | c) with Laplace smoothing.
        smoothed = (self.feature_count_ + self.alpha) / (
            self.class_count_[:, None] + 2 * self.alpha
        )
        self.feature_log_prob_ = np.log(smoothed)
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(
                len(self.classes_), -np.log(len(self.classes_))
            )
        return self

    def _joint_log_likelihood(self, X):
        if self.classes_ is None:
            raise ValueError("BernoulliNB is not fitted yet")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError(
                f"Expected X with {self.feature_log_prob_.shape[1]} features, "
                f"got shape {X.shape}"
            )
        if not np.isin(X, [0, 1]).all():
            raise ValueError("BernoulliNB requires binary features (0 or 1 only)")
        # log P(x | c) = sum_j x_j * log p_jc + (1 - x_j) * log (1 - p_jc)
        neg_prob = np.log1p(-np.exp(self.feature_log_prob_))
        jll = X @ self.feature_log_prob_.T + (1 - X) @ neg_prob.T
        return jll + self.class_log_prior_

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob = jll - np.logaddexp.reduce(jll, axis=1, keepdims=True)
        return log_prob

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
