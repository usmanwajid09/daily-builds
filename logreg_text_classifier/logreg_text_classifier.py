"""Logistic-regression text classifier over TF-IDF features (numpy only).

Public API
----------
TfidfVectorizer
    Sublinear-TF, smoothed-IDF, L2-normalised TF-IDF vectoriser.
LogisticRegressionTextClassifier
    Multiclass logistic regression (softmax + cross-entropy) trained with
    batch gradient descent on TF-IDF features.

Example
-------
>>> from logreg_text_classifier import LogisticRegressionTextClassifier
>>> clf = LogisticRegressionTextClassifier(learning_rate=0.5, epochs=300)
>>> clf.fit(["the striker scored a goal", "bake the bread in the oven"],
...         ["sports", "cooking"])
>>> clf.predict(["the goalkeeper saved the penalty"])
['sports']
"""

import re

import numpy as np

__all__ = ["TfidfVectorizer", "LogisticRegressionTextClassifier", "tokenize"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Split text into lowercase alphanumeric word tokens.

    Parameters
    ----------
    text : str
        The document to tokenise.

    Returns
    -------
    list of str
        Lowercase alphanumeric tokens, in order of appearance.

    Raises
    ------
    TypeError
        If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected a string document, got {type(text).__name__}")
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    """TF-IDF vectoriser with sublinear TF and smoothed IDF.

    TF is ``1 + log(count)`` for terms that occur, IDF is
    ``log((1 + N) / (1 + df)) + 1`` (smoothed, so no term ever gets a
    zero weight), and every row is L2-normalised. The vocabulary is
    built from the training documents only.
    """

    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = None
        self._fitted = False

    def fit(self, docs):
        """Learn the vocabulary and IDF weights from ``docs``.

        Raises
        ------
        ValueError
            If ``docs`` is empty or any document yields no tokens.
        TypeError
            If any document is not a string.
        """
        docs = list(docs)
        if len(docs) == 0:
            raise ValueError("cannot fit TfidfVectorizer on empty training data")
        tokenised = []
        for doc in docs:
            tokens = tokenize(doc)  # raises TypeError for non-strings
            if not tokens:
                raise ValueError("document contains no alphanumeric tokens: "
                                 f"{doc!r}")
            tokenised.append(tokens)

        n_docs = len(tokenised)
        doc_freq = {}
        for tokens in tokenised:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Sorted vocabulary -> deterministic column order.
        self.vocabulary_ = {token: i
                            for i, token in enumerate(sorted(doc_freq))}
        df = np.array([doc_freq[token] for token in sorted(doc_freq)],
                      dtype=float)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self._fitted = True
        return self

    def transform(self, docs):
        """Transform documents into a dense TF-IDF matrix (rows L2-normalised).

        Documents containing only out-of-vocabulary words become all-zero
        rows. Raises RuntimeError if the vectoriser has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("TfidfVectorizer is not fitted yet; call fit() first")
        docs = list(docs)
        n_features = len(self.vocabulary_)
        matrix = np.zeros((len(docs), n_features), dtype=float)
        for row, doc in enumerate(docs):
            counts = {}
            for token in tokenize(doc):
                idx = self.vocabulary_.get(token)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            for idx, count in counts.items():
                matrix[row, idx] = (1.0 + np.log(count)) * self.idf_[idx]
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix

    def fit_transform(self, docs):
        """Fit on ``docs`` and return their TF-IDF matrix."""
        return self.fit(docs).transform(docs)


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class LogisticRegressionTextClassifier:
    """Multiclass logistic regression over TF-IDF features.

    Trains with batch gradient descent on the softmax cross-entropy
    loss with L2 regularisation (the bias is not regularised).

    Parameters
    ----------
    learning_rate : float, default 0.5
        Gradient-descent step size. Must be positive.
    epochs : int, default 300
        Number of full passes over the training data. Must be positive.
    reg : float, default 0.01
        L2 regularisation strength. Must be non-negative.
    random_state : int, default 0
        Seed for the small random weight initialisation, so repeated
        fits on the same data give identical predictions.
    """

    def __init__(self, learning_rate=0.5, epochs=300, reg=0.01, random_state=0):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs}")
        if reg < 0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.reg = float(reg)
        self.random_state = random_state
        self.vectorizer_ = TfidfVectorizer()
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self._fitted = False

    def fit(self, X, y):
        """Fit the vectoriser and the logistic-regression weights.

        Parameters
        ----------
        X : sequence of str
            Training documents.
        y : sequence
            Class label per document.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If ``X`` is a single string, or any document is not a string.
        ValueError
            If the training data is empty, ``X`` and ``y`` have different
            lengths, or any document has no tokens.
        """
        if isinstance(X, str):
            raise TypeError("X must be a sequence of documents, not a single string")
        X = list(X)
        y = list(y)
        if len(X) == 0:
            raise ValueError("cannot fit on empty training data")
        if len(X) != len(y):
            raise ValueError(
                f"X and y have mismatched lengths: {len(X)} documents but "
                f"{len(y)} labels")
        if any(label is None for label in y):
            raise ValueError("labels must not be None")
        if len(set(y)) < 2:
            raise ValueError("need at least two distinct classes to train, "
                             f"got {sorted(set(y), key=str)!r}")

        X_tfidf = self.vectorizer_.fit_transform(X)  # validates docs
        self.classes_ = np.array(sorted(set(y), key=str))
        y_idx = np.array([list(self.classes_).index(label) for label in y])
        n_samples, n_features = X_tfidf.shape
        n_classes = len(self.classes_)

        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.intercept_ = np.zeros(n_classes)

        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y_idx] = 1.0

        for _ in range(self.epochs):
            logits = X_tfidf @ self.coef_ + self.intercept_
            probs = _softmax(logits)
            error = (probs - one_hot) / n_samples
            grad_w = X_tfidf.T @ error + self.reg * self.coef_
            grad_b = error.sum(axis=0)
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "LogisticRegressionTextClassifier is not fitted yet; "
                "call fit() before predict/predict_proba")

    def _vectorise(self, X):
        self._check_fitted()
        if isinstance(X, str):
            raise TypeError("X must be a sequence of documents, not a single string")
        X = list(X)
        if len(X) == 0:
            raise ValueError("cannot predict on empty input")
        for doc in X:
            if not isinstance(doc, str):
                raise TypeError(
                    f"expected string documents, got {type(doc).__name__}")
            if not doc.strip():
                raise ValueError("cannot predict on an empty document")
        return self.vectorizer_.transform(X)

    def predict_proba(self, X):
        """Return per-class probabilities; every row sums to 1."""
        X_tfidf = self._vectorise(X)
        logits = X_tfidf @ self.coef_ + self.intercept_
        return _softmax(logits)

    def predict(self, X):
        """Return the most probable class label for each document."""
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return [self.classes_[i] for i in indices]

    def score(self, X, y):
        """Return classification accuracy on ``(X, y)``."""
        predictions = self.predict(X)
        return float(np.mean([p == t for p, t in zip(predictions, y)]))
