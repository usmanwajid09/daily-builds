"""Latent Semantic Analysis (LSA) topic modelling.

TF-IDF -> truncated SVD -> topics as term loadings.

Only numpy is used: TF-IDF is built by hand and the truncated SVD is
``numpy.linalg.svd`` on the (documents x terms) TF-IDF matrix.
"""

import re

import numpy as np

__all__ = ["LSATopicModel", "tokenize"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small built-in stopword list. Function words occur in every theme and
# would otherwise dominate the top singular vectors, burying the real
# topics; dropping them is standard practice for topic modelling.
STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about
    into through after over under again further once here there all any
    both each few more most other some such no nor not only own same so
    than too very can will just should now is am are was were be been
    being have has had having do does did doing would could shall may
    might must i you he she it we they me him her us them my your his
    its our their this that these those as from on in to out up down off
    """.split()
)


def tokenize(text):
    """Lowercase alphanumeric tokeniser (stopwords removed).

    Parameters
    ----------
    text : str
        A single document.

    Returns
    -------
    list of str
        Lowercase alphanumeric tokens, in order of appearance, with
        common English stopwords removed.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected a string document, got {type(text).__name__}")
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


class LSATopicModel:
    """LSA topic model: TF-IDF followed by truncated SVD.

    Each latent topic is a row of the right singular vectors ``V``: a
    loading over the vocabulary. Documents are represented by their
    projection onto the top ``n_topics`` singular vectors.

    Parameters
    ----------
    n_topics : int
        Number of latent topics (singular vectors) to keep. Must satisfy
        ``1 <= n_topics < min(n_documents, n_terms)`` once fitted.

    Attributes
    ----------
    vocabulary_ : list of str
        Terms, in column order of the TF-IDF matrix.
    components_ : ndarray, shape (n_topics, n_terms)
        Topic-term loadings (rows of ``V`` from the truncated SVD).
    singular_values_ : ndarray, shape (n_topics,)
        Top singular values of the TF-IDF matrix.
    explained_variance_ratio_ : ndarray, shape (n_topics,)
        ``singular_values_**2`` normalised by the total squared
        singular values, so the entries sum to at most 1.
    tfidf_ : ndarray, shape (n_documents, n_terms)
        The TF-IDF matrix the model was fitted on.
    """

    def __init__(self, n_topics=2):
        if isinstance(n_topics, bool) or not isinstance(n_topics, (int, np.integer)):
            raise TypeError(
                f"n_topics must be an integer, got {type(n_topics).__name__}"
            )
        if n_topics < 1:
            raise ValueError(f"n_topics must be >= 1, got {n_topics}")
        self.n_topics = int(n_topics)
        self.vocabulary_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.tfidf_ = None
        self._doc_term_counts = None  # for coherence (document frequencies)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError(
                "LSATopicModel is not fitted yet; call fit(documents) first."
            )

    @staticmethod
    def _validate_documents(documents):
        if documents is None:
            raise TypeError("documents must be a sequence of strings, got None")
        if isinstance(documents, (str, bytes)):
            raise TypeError(
                "documents must be a sequence of strings, not a single string"
            )
        try:
            documents = list(documents)
        except TypeError:
            raise TypeError(
                "documents must be a sequence of strings, "
                f"got {type(documents).__name__}"
            ) from None
        if len(documents) == 0:
            raise ValueError("documents is empty; provide at least one document")
        for doc in documents:
            if not isinstance(doc, str):
                raise TypeError(
                    f"each document must be a string, got {type(doc).__name__}"
                )
        return documents

    def _tfidf(self, documents):
        """Build the TF-IDF matrix for ``documents`` using the fitted vocab.

        TF is sublinear (``1 + log(count)``), IDF is smoothed
        (``log((1 + N) / (1 + df)) + 1``), rows are L2-normalised.
        """
        n_docs = len(documents)
        n_terms = len(self.vocabulary_)
        index = {term: j for j, term in enumerate(self.vocabulary_)}
        counts = np.zeros((n_docs, n_terms), dtype=float)
        for i, doc in enumerate(documents):
            for tok in tokenize(doc):
                j = index.get(tok)
                if j is not None:
                    counts[i, j] += 1.0
        tf = np.zeros_like(counts)
        _mask = counts > 0
        tf[_mask] = 1.0 + np.log(counts[_mask])
        idf = self._idf  # fitted on the training corpus
        tfidf = tf * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # all-OOV documents stay a zero row
        return tfidf / norms

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, documents):
        """Fit the TF-IDF vectoriser and truncated SVD.

        Parameters
        ----------
        documents : sequence of str
            The training corpus.

        Returns
        -------
        self
        """
        documents = self._validate_documents(documents)
        tokenised = [tokenize(doc) for doc in documents]
        if all(len(toks) == 0 for toks in tokenised):
            raise ValueError(
                "documents contain no alphanumeric tokens; "
                "provide documents with real words"
            )

        vocab = sorted({tok for toks in tokenised for tok in toks})
        n_docs = len(documents)
        n_terms = len(vocab)
        if self.n_topics >= min(n_docs, n_terms):
            raise ValueError(
                f"n_topics={self.n_topics} must be < min(n_documents, n_terms)"
                f" = min({n_docs}, {n_terms}) = {min(n_docs, n_terms)}; "
                "use fewer topics or provide more documents/terms"
            )

        self.vocabulary_ = vocab
        index = {term: j for j, term in enumerate(vocab)}
        counts = np.zeros((n_docs, n_terms), dtype=float)
        df = np.zeros(n_terms, dtype=float)
        for i, toks in enumerate(tokenised):
            seen = set()
            for tok in toks:
                j = index[tok]
                counts[i, j] += 1.0
                seen.add(j)
            for j in seen:
                df[j] += 1.0

        tf = np.zeros_like(counts)
        _mask = counts > 0
        tf[_mask] = 1.0 + np.log(counts[_mask])
        # Smoothed IDF: log((1 + N) / (1 + df)) + 1
        self._idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        tfidf = tf * self._idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        tfidf = tfidf / norms
        self.tfidf_ = tfidf
        self._doc_term_counts = (counts > 0)

        # Truncated SVD of the (documents x terms) TF-IDF matrix.
        _, s, vt = np.linalg.svd(tfidf, full_matrices=False)
        self.singular_values_ = s[: self.n_topics]
        self.components_ = vt[: self.n_topics]
        total = float(np.sum(s**2))
        self.explained_variance_ratio_ = (
            (self.singular_values_**2) / total if total > 0 else np.zeros(self.n_topics)
        )
        return self

    # ------------------------------------------------------------------
    # topics
    # ------------------------------------------------------------------
    def get_topics(self, n_words=5):
        """Return the top words of each topic.

        Parameters
        ----------
        n_words : int
            Words per topic. Clamped to the vocabulary size.

        Returns
        -------
        list of list of (str, float)
            One list per topic, each a list of ``(term, weight)`` pairs
            sorted by ``|weight|`` descending. Weights are the term
            loadings of the topic's singular vector (sign is arbitrary,
            magnitude is what matters).
        """
        self._check_fitted()
        if isinstance(n_words, bool) or not isinstance(n_words, (int, np.integer)):
            raise TypeError(
                f"n_words must be an integer, got {type(n_words).__name__}"
            )
        if n_words < 1:
            raise ValueError(f"n_words must be >= 1, got {n_words}")
        n_words = min(int(n_words), len(self.vocabulary_))
        topics = []
        for row in self.components_:
            order = np.argsort(-np.abs(row))[:n_words]
            topics.append(
                [(self.vocabulary_[j], float(row[j])) for j in order]
            )
        return topics

    def transform(self, documents):
        """Project documents into topic space.

        Parameters
        ----------
        documents : sequence of str
            Documents to project (need not be the training documents).

        Returns
        -------
        ndarray, shape (n_documents, n_topics)
            Document-topic loadings. SVD signs are arbitrary, so compare
            topics by magnitude (``abs``) when assigning a dominant topic.
        """
        self._check_fitted()
        documents = self._validate_documents(documents)
        return self._tfidf(documents) @ self.components_.T

    def dominant_topics(self, documents):
        """Dominant topic index of each document (by |loading|).

        Returns
        -------
        ndarray, shape (n_documents,)
            Index of the topic with the largest absolute loading.
        """
        return np.argmax(np.abs(self.transform(documents)), axis=1)

    def reconstruction_error(self, documents=None):
        """Frobenius reconstruction error of the TF-IDF matrix.

        Reconstructs TF-IDF as ``transform(documents) @ components_`` and
        returns ``||X - X_hat||_F / ||X||_F``. With ``documents=None``
        the training matrix is used. The error falls as ``n_topics``
        grows.
        """
        self._check_fitted()
        if documents is None:
            x = self.tfidf_
            approx = (x @ self.components_.T) @ self.components_
        else:
            documents = self._validate_documents(documents)
            x = self._tfidf(documents)
            approx = self.transform(documents) @ self.components_
        denom = np.linalg.norm(x)
        if denom == 0.0:
            return 0.0
        return float(np.linalg.norm(x - approx) / denom)

    def topic_coherence(self, n_words=5):
        """UMass-style coherence of each topic on the training corpus.

        For a topic's top words, the mean over word pairs ``(w_i, w_j)``,
        ``i < j``, of ``log((D(w_i, w_j) + eps) / D(w_j))``, where
        ``D`` counts training documents containing the word(s). Higher
        (closer to zero) means the top words co-occur more, i.e. a more
        coherent topic.

        Returns
        -------
        list of float
            One coherence score per topic.
        """
        self._check_fitted()
        if isinstance(n_words, bool) or not isinstance(n_words, (int, np.integer)):
            raise TypeError(
                f"n_words must be an integer, got {type(n_words).__name__}"
            )
        if n_words < 1:
            raise ValueError(f"n_words must be >= 1, got {n_words}")
        n_words = min(int(n_words), len(self.vocabulary_))
        eps = 1e-12
        doc_occurrence = self._doc_term_counts  # (n_docs, n_terms) bool
        scores = []
        for row in self.components_:
            top = np.argsort(-np.abs(row))[:n_words]
            pair_scores = []
            for a in range(1, len(top)):
                for b in range(a):
                    both = float(np.sum(doc_occurrence[:, top[a]] & doc_occurrence[:, top[b]]))
                    only_b = float(np.sum(doc_occurrence[:, top[b]]))
                    pair_scores.append(np.log((both + eps) / (only_b + eps)))
            scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
        return scores
