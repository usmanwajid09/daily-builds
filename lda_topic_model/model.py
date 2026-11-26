"""Latent Dirichlet Allocation topic model, from scratch.

Collapsed Gibbs sampling over a bag-of-words corpus: each document is a
mixture of topics and each topic is a distribution over words. Only
numpy (and the standard library) is used.
"""

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text):
    """Lowercase a string and return its alphabetic word tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _TOKEN_RE.findall(text.lower())


class LatentDirichletAllocation:
    """LDA topic model fitted with collapsed Gibbs sampling.

    Parameters
    ----------
    n_topics : int
        Number of topics to discover. Must be >= 2.
    alpha : float
        Dirichlet prior on document-topic distributions. Must be > 0.
        Smaller values make documents concentrate on fewer topics.
    beta : float
        Dirichlet prior on topic-word distributions. Must be > 0.
        Smaller values make topics concentrate on fewer words.
    n_iter : int
        Number of Gibbs sampling sweeps over the corpus. Must be >= 1.
    random_state : int or None
        Seed for the sampler; a fixed seed makes fits reproducible.
    """

    def __init__(self, n_topics=2, alpha=0.1, beta=0.1, n_iter=200,
                 random_state=42):
        if not isinstance(n_topics, (int, np.integer)):
            raise TypeError("n_topics must be an integer")
        if n_topics < 2:
            raise ValueError("n_topics must be at least 2")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if not isinstance(n_iter, (int, np.integer)):
            raise TypeError("n_iter must be an integer")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if random_state is not None and not isinstance(
                random_state, (int, np.integer)):
            raise TypeError("random_state must be an integer or None")
        self.n_topics = int(n_topics)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n_iter = int(n_iter)
        self.random_state = random_state
        self.vocab_ = None
        self.topic_word_dist_ = None
        self.doc_topic_dist_ = None

    def _require_fit(self):
        if self.vocab_ is None:
            raise ValueError(
                "LatentDirichletAllocation is not fitted yet; call fit() first")

    def fit(self, documents):
        """Fit the model on a list of document strings. Returns self."""
        docs = list(documents)
        if len(docs) == 0:
            raise ValueError("at least one document is required")
        tokenised = []
        for doc in docs:
            if not isinstance(doc, str):
                raise TypeError("every document must be a string")
            tokens = tokenize(doc)
            if len(tokens) == 0:
                raise ValueError(
                    "every document must contain at least one word token")
            tokenised.append(tokens)

        vocab = sorted({tok for doc in tokenised for tok in doc})
        self.vocab_ = vocab
        word_index = {w: i for i, w in enumerate(vocab)}
        n_docs = len(docs)
        n_vocab = len(vocab)
        K = self.n_topics

        # Token ids per document.
        doc_tokens = [[word_index[t] for t in doc] for doc in tokenised]

        rng = np.random.default_rng(self.random_state)
        n_dk = np.zeros((n_docs, K), dtype=np.int64)   # doc-topic counts
        n_kw = np.zeros((K, n_vocab), dtype=np.int64)  # topic-word counts
        n_k = np.zeros(K, dtype=np.int64)              # topic totals
        z = []  # topic assignment per token
        for d, tokens in enumerate(doc_tokens):
            assignments = rng.integers(0, K, size=len(tokens))
            z.append(assignments)
            for w, k in zip(tokens, assignments):
                n_dk[d, k] += 1
                n_kw[k, w] += 1
                n_k[k] += 1

        # Collapsed Gibbs sampling sweeps.
        for _ in range(self.n_iter):
            for d, tokens in enumerate(doc_tokens):
                for i, w in enumerate(tokens):
                    k_old = z[d][i]
                    n_dk[d, k_old] -= 1
                    n_kw[k_old, w] -= 1
                    n_k[k_old] -= 1
                    # P(topic k | rest) ∝ (n_dk + alpha) * (n_kw + beta) / (n_k + V*beta)
                    p = ((n_dk[d] + self.alpha)
                         * (n_kw[:, w] + self.beta)
                         / (n_k + n_vocab * self.beta))
                    p = p / p.sum()
                    k_new = int(rng.choice(K, p=p))
                    z[d][i] = k_new
                    n_dk[d, k_new] += 1
                    n_kw[k_new, w] += 1
                    n_k[k_new] += 1

        self.topic_word_dist_ = ((n_kw + self.beta)
                                 / (n_k[:, None] + n_vocab * self.beta))
        self.doc_topic_dist_ = ((n_dk + self.alpha)
                                / (n_dk.sum(axis=1, keepdims=True)
                                   + K * self.alpha))
        return self

    def top_words(self, topic, n=10):
        """The n most probable words of one topic, highest first."""
        self._require_fit()
        if not isinstance(topic, (int, np.integer)):
            raise TypeError("topic must be an integer index")
        if topic < 0 or topic >= self.n_topics:
            raise ValueError(
                f"topic must be between 0 and {self.n_topics - 1}")
        if not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer")
        if n < 1:
            raise ValueError("n must be at least 1")
        order = np.argsort(-self.topic_word_dist_[topic])[:n]
        return [self.vocab_[i] for i in order]

    def doc_topic_dist(self, doc_index):
        """Topic distribution of one fitted document (sums to 1)."""
        self._require_fit()
        if not isinstance(doc_index, (int, np.integer)):
            raise TypeError("doc_index must be an integer index")
        if doc_index < 0 or doc_index >= self.doc_topic_dist_.shape[0]:
            raise ValueError(
                f"doc_index must be between 0 and "
                f"{self.doc_topic_dist_.shape[0] - 1}")
        return self.doc_topic_dist_[doc_index].copy()

    def transform(self, documents):
        """Topic distributions for new documents (fold-in Gibbs sampling).

        The fitted topic-word distributions are held fixed while topic
        assignments for the new documents are sampled. Returns an array
        of shape (n_documents, n_topics) whose rows sum to 1.
        """
        self._require_fit()
        docs = list(documents)
        if len(docs) == 0:
            raise ValueError("at least one document is required")
        word_index = {w: i for i, w in enumerate(self.vocab_)}
        K = self.n_topics
        rng = np.random.default_rng(self.random_state)
        out = np.zeros((len(docs), K), dtype=float)
        for d, doc in enumerate(docs):
            if not isinstance(doc, str):
                raise TypeError("every document must be a string")
            tokens = [word_index[t] for t in tokenize(doc)
                      if t in word_index]
            if len(tokens) == 0:
                # No known words: fall back to the prior (uniform).
                out[d] = 1.0 / K
                continue
            assignments = rng.integers(0, K, size=len(tokens))
            n_dk = np.zeros(K, dtype=np.int64)
            for k in assignments:
                n_dk[k] += 1
            for _ in range(100):
                for i, w in enumerate(tokens):
                    k_old = assignments[i]
                    n_dk[k_old] -= 1
                    p = ((n_dk + self.alpha)
                         * self.topic_word_dist_[:, w])
                    p = p / p.sum()
                    k_new = int(rng.choice(K, p=p))
                    assignments[i] = k_new
                    n_dk[k_new] += 1
            out[d] = (n_dk + self.alpha) / (len(tokens) + K * self.alpha)
        return out
