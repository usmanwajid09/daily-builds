"""TF-IDF vectorizer plus cosine-similarity search, standard library only.

Documents are tokenised to lowercase alphanumeric words. Term frequency
uses ``1 + log(count)`` for counts above zero; inverse document frequency
uses the smoothed form ``log((1 + N) / (1 + df)) + 1``. Vectors are
L2-normalised, so cosine similarity is a plain dot product.
"""

import math
import re

__all__ = ["TfidfVectorizer"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class TfidfVectorizer:
    """Fit a TF-IDF model on a small document set and search it."""

    def __init__(self):
        self.vocabulary_ = []
        self.idf_ = {}
        self._doc_vectors = []
        self._fitted = False

    def fit(self, documents):
        """Learn the vocabulary and IDF weights from ``documents``.

        ``documents`` must be a non-empty list of non-empty strings.
        Returns ``self``.
        """
        if isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a list of strings, got a string")
        try:
            documents = list(documents)
        except TypeError:
            raise TypeError("documents must be a list of strings")
        if len(documents) == 0:
            raise ValueError("documents must not be empty")
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise TypeError(
                    f"document {i} must be a string, got {type(doc).__name__!r}"
                )
            if doc.strip() == "":
                raise ValueError(f"document {i} must not be empty or blank")

        tokenized = [_tokenize(doc) for doc in documents]
        df = {}
        for tokens in tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        n_docs = len(documents)
        self.vocabulary_ = sorted(df)
        self.idf_ = {
            term: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for term, count in df.items()
        }
        self._doc_vectors = [self._vectorize(tokens) for tokens in tokenized]
        self._fitted = True
        return self

    def _vectorize(self, tokens):
        counts = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        vector = {}
        for term, count in counts.items():
            if term in self.idf_:
                vector[term] = (1.0 + math.log(count)) * self.idf_[term]
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm == 0.0:
            return {}
        return {term: value / norm for term, value in vector.items()}

    def transform(self, documents):
        """Return TF-IDF vectors (list of ``{term: weight}`` dicts)."""
        if not self._fitted:
            raise RuntimeError("vectorizer is not fitted yet; call fit() first")
        if isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a list of strings, got a string")
        try:
            documents = list(documents)
        except TypeError:
            raise TypeError("documents must be a list of strings")
        if len(documents) == 0:
            raise ValueError("documents must not be empty")
        vectors = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise TypeError(
                    f"document {i} must be a string, got {type(doc).__name__!r}"
                )
            vectors.append(self._vectorize(_tokenize(doc)))
        return vectors

    def fit_transform(self, documents):
        """Fit on ``documents`` and return their TF-IDF vectors."""
        self.fit(documents)
        return [dict(vector) for vector in self._doc_vectors]

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """Cosine similarity of two sparse ``{term: weight}`` vectors."""
        if not isinstance(vec_a, dict) or not isinstance(vec_b, dict):
            raise TypeError("both vectors must be dicts")
        if len(vec_a) == 0 or len(vec_b) == 0:
            return 0.0
        smaller, larger = (vec_a, vec_b) if len(vec_a) <= len(vec_b) else (vec_b, vec_a)
        return sum(weight * larger.get(term, 0.0) for term, weight in smaller.items())

    def search(self, query, top_k=3):
        """Rank the fitted documents by cosine similarity to ``query``.

        Returns a list of ``(doc_index, score)`` pairs, best first, with
        at most ``top_k`` entries. Documents scoring 0 are still returned
        (they rank last) so the caller can see the full ranking.
        """
        if not self._fitted:
            raise RuntimeError("vectorizer is not fitted yet; call fit() first")
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__!r}")
        if query.strip() == "":
            raise ValueError("query must not be empty or blank")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k!r}")

        query_vector = self._vectorize(_tokenize(query))
        scored = [
            (i, self.cosine_similarity(query_vector, doc_vector))
            for i, doc_vector in enumerate(self._doc_vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
