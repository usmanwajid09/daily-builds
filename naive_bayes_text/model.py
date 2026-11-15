"""Multinomial Naive Bayes text classifier, from scratch.

Tokens are lowercased words; each class gets a prior and a smoothed
word likelihood (Laplace smoothing). Prediction uses log-probabilities,
so long documents never underflow. Only numpy (and the standard
library) is used.
"""

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text):
    """Lowercase a string and return its alphabetic word tokens."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return _TOKEN_RE.findall(text.lower())


class MultinomialNB:
    """Multinomial Naive Bayes classifier for text.

    Parameters
    ----------
    alpha : float
        Laplace smoothing strength. Must be > 0.
    """

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.classes_ = None
        self.vocab_ = None
        self._log_prior = None
        self._log_likelihood = None

    def fit(self, texts, labels):
        """Fit on a list of documents and their labels. Returns self."""
        texts = list(texts)
        labels = list(labels)
        if len(texts) == 0:
            raise ValueError("at least one training document is required")
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        tokenised = [tokenize(t) for t in texts]
        if all(len(t) == 0 for t in tokenised):
            raise ValueError("training documents contain no tokens")

        self.classes_ = np.array(sorted(set(labels)))
        class_index = {c: i for i, c in enumerate(self.classes_)}
        vocab = sorted({tok for doc in tokenised for tok in doc})
        self.vocab_ = vocab
        word_index = {w: i for i, w in enumerate(vocab)}
        n_classes = len(self.classes_)
        counts = np.zeros((n_classes, len(vocab)), dtype=float)
        class_doc_counts = np.zeros(n_classes, dtype=float)
        for doc, label in zip(tokenised, labels):
            ci = class_index[label]
            class_doc_counts[ci] += 1
            for tok in doc:
                counts[ci, word_index[tok]] += 1

        self._log_prior = np.log(class_doc_counts / len(texts))
        smoothed = counts + self.alpha
        self._log_likelihood = np.log(
            smoothed / smoothed.sum(axis=1, keepdims=True)
        )
        return self

    def _require_fit(self):
        if self.classes_ is None:
            raise ValueError("MultinomialNB is not fitted yet; call fit() first")

    def _doc_matrix(self, texts):
        self._require_fit()
        texts = list(texts)
        if len(texts) == 0:
            raise ValueError("at least one document is required")
        word_index = {w: i for i, w in enumerate(self.vocab_)}
        mat = np.zeros((len(texts), len(self.vocab_)), dtype=float)
        for r, text in enumerate(texts):
            for tok in tokenize(text):
                if tok in word_index:
                    mat[r, word_index[tok]] += 1
        return mat

    def predict_log_proba(self, texts):
        """Log-probability of each class for each document."""
        mat = self._doc_matrix(texts)
        return mat @ self._log_likelihood.T + self._log_prior

    def predict_proba(self, texts):
        """Class probabilities for each document (rows sum to 1)."""
        log_p = self.predict_log_proba(texts)
        shifted = log_p - log_p.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, texts):
        """Predicted label for each document."""
        log_p = self.predict_log_proba(texts)
        return self.classes_[np.argmax(log_p, axis=1)]

    def score(self, texts, labels):
        """Classification accuracy on labelled documents."""
        preds = self.predict(texts)
        labels = np.asarray(labels)
        if len(labels) != len(preds):
            raise ValueError("labels must have one entry per document")
        return float(np.mean(preds == labels))
