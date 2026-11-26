"""Skip-gram word embeddings (word2vec) from scratch, with negative sampling.

Each word gets a dense vector. Training slides a window over each
sentence and asks the model to predict the words around a centre word
(the "skip-gram" objective). Instead of a costly softmax over the whole
vocabulary, each real (centre, context) pair is trained against a few
randomly drawn "negative" words using binary logistic loss, and the
vectors are updated with plain SGD. Words that appear in similar
contexts end up with similar vectors, so cosine similarity between
vectors measures how alike two words are.

Only numpy and the standard library are used.
"""

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+")

# Very common words carry almost no meaning about *which* word they sit
# next to (nearly every word follows "the"), so they are dropped when
# building training pairs. They are not dropped by tokenize() itself.
STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "in", "my", "i", "and", "to", "of",
        "at", "on", "for", "with", "are", "was", "were", "it", "its",
    }
)


def _stem(word):
    """Tiny stemmer: strip a trailing plural "s" ("eats" -> "eat").

    This keeps verb/noun forms that mean the same thing in one
    vocabulary entry, which matters on a tiny corpus where "eats" and
    "eat" would otherwise split a word's contexts in two.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text):
    """Lowercase a string and return its word tokens (stemmed)."""
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text).__name__}")
    return [_stem(tok) for tok in _TOKEN_RE.findall(text.lower())]


def _sigmoid(x):
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def _check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


class SkipGramWord2Vec:
    """Skip-gram word2vec trained with negative sampling and SGD.

    Parameters
    ----------
    embedding_dim : int
        Size of each word vector. Must be >= 1.
    window : int
        How many words to each side of a centre word count as context.
        Must be >= 1.
    negative_samples : int
        Number of random negative words drawn per real (centre, context)
        pair. Must be >= 1.
    learning_rate : float
        SGD step size. Must be > 0.
    epochs : int
        Full passes over all (centre, context) pairs. Must be >= 1.
    min_count : int
        Words appearing fewer times than this are dropped from the
        vocabulary. Must be >= 1.
    seed : int or None
        Random seed for reproducibility. Two models fitted with the same
        seed on the same sentences produce identical embeddings.
    """

    def __init__(
        self,
        embedding_dim=16,
        window=2,
        negative_samples=5,
        learning_rate=0.05,
        epochs=200,
        min_count=1,
        seed=42,
    ):
        _check_int("embedding_dim", embedding_dim, 1)
        _check_int("window", window, 1)
        _check_int("negative_samples", negative_samples, 1)
        _check_int("epochs", epochs, 1)
        _check_int("min_count", min_count, 1)
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise TypeError(
                f"learning_rate must be a number, got {learning_rate!r}"
            )
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}"
            )
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError(f"seed must be an integer or None, got {seed!r}")

        self.embedding_dim = int(embedding_dim)
        self.window = int(window)
        self.negative_samples = int(negative_samples)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.seed = seed

        self.vocab_ = None
        self.word_to_index_ = None
        self.embeddings_ = None
        self.loss_history_ = None
        self.n_pairs_ = None
        self._embeddings_norm = None

    def __contains__(self, word):
        return self.vocab_ is not None and word in self.word_to_index_

    def _require_fit(self):
        if self.vocab_ is None:
            raise ValueError(
                "SkipGramWord2Vec is not fitted yet; call fit() first"
            )

    def fit(self, sentences):
        """Build the vocabulary and train the embeddings. Returns self.

        ``sentences`` is a list of strings, or a list of token lists
        (each token a string). Stopwords are dropped before training.
        """
        if isinstance(sentences, (str, bytes)) or not isinstance(
            sentences, (list, tuple)
        ):
            raise TypeError(
                "sentences must be a list of strings (or a list of token "
                f"lists), got {type(sentences).__name__}"
            )
        if len(sentences) == 0:
            raise ValueError("at least one sentence is required")
        tokenised = []
        for i, sent in enumerate(sentences):
            if isinstance(sent, str):
                toks = tokenize(sent)
            elif isinstance(sent, (list, tuple)):
                if not all(isinstance(t, str) for t in sent):
                    raise ValueError(
                        f"sentence {i} contains a non-string token; every "
                        "token must be a string"
                    )
                toks = tokenize(" ".join(sent))
            else:
                raise TypeError(
                    "sentence "
                    f"{i} must be a string or a list of tokens, got "
                    f"{type(sent).__name__}"
                )
            tokenised.append([t for t in toks if t not in STOPWORDS])
        if all(len(t) == 0 for t in tokenised):
            raise ValueError(
                "sentences contain no usable tokens after removing "
                "stopwords; pass sentences with real content words"
            )

        counts = {}
        for toks in tokenised:
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
        vocab = sorted(w for w, c in counts.items() if c >= self.min_count)
        if len(vocab) < 2:
            raise ValueError(
                f"vocabulary has only {len(vocab)} word(s) after "
                f"min_count={self.min_count}; need at least 2 distinct "
                "words to learn similarities"
            )
        self.vocab_ = vocab
        self.word_to_index_ = {w: i for i, w in enumerate(vocab)}
        vocab_size = len(vocab)

        pairs = []
        for toks in tokenised:
            ids = [self.word_to_index_[t] for t in toks if t in self.word_to_index_]
            for i, centre in enumerate(ids):
                for j in range(
                    max(0, i - self.window),
                    min(len(ids), i + self.window + 1),
                ):
                    if j != i:
                        pairs.append((centre, ids[j]))
        if len(pairs) == 0:
            raise ValueError(
                "no co-occurrence pairs could be built: every sentence "
                "has fewer than 2 vocabulary words after stopword removal; "
                "use longer sentences or a larger window"
            )
        pairs = np.array(pairs, dtype=int)
        self.n_pairs_ = len(pairs)

        rng = np.random.default_rng(self.seed)
        # Small random start for input vectors; output vectors start at 0,
        # the standard word2vec initialisation.
        embeddings = (rng.random((vocab_size, self.embedding_dim)) - 0.5) / (
            self.embedding_dim
        )
        output = np.zeros((vocab_size, self.embedding_dim), dtype=float)

        # Negative words are drawn from the unigram distribution raised to
        # 3/4, which down-weights very frequent words (Mikolov et al.).
        freq = np.array([counts[w] for w in vocab], dtype=float)
        neg_probs = freq ** 0.75
        neg_probs /= neg_probs.sum()

        lr = self.learning_rate
        n_neg = self.negative_samples
        loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(pairs))
            # Draw all negatives for the epoch in one vectorised call.
            negatives = rng.choice(
                vocab_size, size=(len(pairs), n_neg), p=neg_probs
            )
            epoch_loss = 0.0
            epoch_samples = 0
            for idx in order:
                centre = int(pairs[idx, 0])
                context = int(pairs[idx, 1])
                u = embeddings[centre].copy()
                targets = [context] + [
                    int(n) for n in negatives[idx]
                ]
                for k, target in enumerate(targets):
                    if target == centre:
                        continue
                    label = 1 if k == 0 else 0
                    v = output[target].copy()
                    sig = _sigmoid(float(u @ v))
                    clipped = min(max(sig, 1e-7), 1.0 - 1e-7)
                    epoch_loss += (
                        -np.log(clipped)
                        if label == 1
                        else -np.log(1.0 - clipped)
                    )
                    epoch_samples += 1
                    g = (sig - label) * lr
                    output[target] = v - g * u
                    u = u - g * v
                embeddings[centre] = u
            loss_history.append(epoch_loss / max(epoch_samples, 1))

        self.embeddings_ = embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings_norm = embeddings / norms
        self.loss_history_ = loss_history
        return self

    def get_vector(self, word):
        """Return a copy of the embedding vector for ``word``."""
        self._require_fit()
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {word!r}")
        if word not in self.word_to_index_:
            raise ValueError(
                f"word '{word}' is not in the vocabulary "
                f"(vocabulary: {', '.join(self.vocab_)})"
            )
        return self.embeddings_[self.word_to_index_[word]].copy()

    def similarity(self, word1, word2):
        """Cosine similarity between two words, in [-1, 1]."""
        self._require_fit()
        for name, word in (("word1", word1), ("word2", word2)):
            if not isinstance(word, str):
                raise TypeError(f"{name} must be a string, got {word!r}")
            if word not in self.word_to_index_:
                raise ValueError(
                    f"word '{word}' is not in the vocabulary "
                    f"(vocabulary: {', '.join(self.vocab_)})"
                )
        a = self._embeddings_norm[self.word_to_index_[word1]]
        b = self._embeddings_norm[self.word_to_index_[word2]]
        return float(a @ b)

    def most_similar(self, word, topn=5):
        """``topn`` nearest words to ``word`` as (word, cosine) pairs.

        The query word itself is never returned.
        """
        self._require_fit()
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {word!r}")
        if word not in self.word_to_index_:
            raise ValueError(
                f"word '{word}' is not in the vocabulary "
                f"(vocabulary: {', '.join(self.vocab_)})"
            )
        if isinstance(topn, bool) or not isinstance(topn, int):
            raise TypeError(f"topn must be an integer, got {topn!r}")
        if topn < 1:
            raise ValueError(f"topn must be at least 1, got {topn}")
        sims = self._embeddings_norm @ self._embeddings_norm[
            self.word_to_index_[word]
        ]
        order = np.argsort(-sims)
        query_idx = self.word_to_index_[word]
        return [
            (self.vocab_[i], float(sims[i]))
            for i in order
            if i != query_idx
        ][:topn]

    def analogy(self, word_a, word_b, word_c, topn=5):
        """Solve ``word_a`` is to ``word_b`` as ``?`` is to ``word_c``.

        Returns the words nearest to ``vec(a) - vec(b) + vec(c)``, so
        ``analogy("king", "man", "woman")`` should return "queen" on a
        corpus where that relation holds. ``word_a``, ``word_b`` and
        ``word_c`` themselves are never returned.
        """
        self._require_fit()
        for name, word in (
            ("word_a", word_a),
            ("word_b", word_b),
            ("word_c", word_c),
        ):
            if not isinstance(word, str):
                raise TypeError(f"{name} must be a string, got {word!r}")
            if word not in self.word_to_index_:
                raise ValueError(
                    f"word '{word}' is not in the vocabulary "
                    f"(vocabulary: {', '.join(self.vocab_)})"
                )
        if isinstance(topn, bool) or not isinstance(topn, int):
            raise TypeError(f"topn must be an integer, got {topn!r}")
        if topn < 1:
            raise ValueError(f"topn must be at least 1, got {topn}")
        vec = (
            self._embeddings_norm[self.word_to_index_[word_a]]
            - self._embeddings_norm[self.word_to_index_[word_b]]
            + self._embeddings_norm[self.word_to_index_[word_c]]
        )
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError(
                "analogy vector has zero length; the three words cancel "
                "each other out"
            )
        vec = vec / norm
        sims = self._embeddings_norm @ vec
        excluded = {
            self.word_to_index_[word_a],
            self.word_to_index_[word_b],
            self.word_to_index_[word_c],
        }
        order = np.argsort(-sims)
        return [
            (self.vocab_[i], float(sims[i]))
            for i in order
            if i not in excluded
        ][:topn]
