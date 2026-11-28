"""Byte-pair encoding (BPE) tokenizer, standard library only.

Classic Sennrich-style word-level BPE: each word starts as its individual
characters plus an end-of-word marker, and the most frequent adjacent symbol
pair is repeatedly merged until the requested number of merge rules is
learned (or no pair is frequent enough). Encoding then applies the learned
merges greedily, in learned order, to each word.
"""

import re
from collections import Counter

END_OF_WORD = "</w>"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalise_words(text):
    """Lowercase text and split it into alphanumeric words."""
    return _WORD_RE.findall(text.lower())


def _word_to_symbols(word):
    """A word as a tuple of its characters plus the end-of-word marker."""
    return tuple(word) + (END_OF_WORD,)


def _merge_word(symbols, pair):
    """Merge every adjacent occurrence of ``pair`` in a symbol tuple."""
    first, second = pair
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
            merged.append(first + second)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def _pair_counts(word_freqs):
    """Count adjacent symbol pairs over all words, weighted by frequency.

    Pairs are inserted in first-seen order (words in first-appearance order,
    pairs left to right), so ties are broken deterministically.
    """
    counts = {}
    for symbols, freq in word_freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts


class BPETokenizer:
    """Learn BPE merge rules from a corpus, then encode/decode text.

    Parameters
    ----------
    num_merges : int
        Maximum number of merge rules to learn (must be >= 1).
    min_frequency : int
        A pair must occur at least this many times (counting word
        frequencies) to be merged (must be >= 1).
    """

    def __init__(self, num_merges=50, min_frequency=2):
        if not isinstance(num_merges, int) or isinstance(num_merges, bool):
            raise TypeError("num_merges must be an int")
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool):
            raise TypeError("min_frequency must be an int")
        if num_merges < 1:
            raise ValueError("num_merges must be >= 1")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.merges = []          # list of (left, right) tuples, in learned order
        self.vocab = set()        # set of all symbols learned
        self._fitted = False

    # -- training ---------------------------------------------------------
    def fit(self, corpus):
        """Learn merge rules from ``corpus`` (a string or a list of strings).

        Text is normalised to lowercase alphanumeric words. Returns self.
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        if not isinstance(corpus, (list, tuple)):
            raise TypeError("corpus must be a string or a list of strings")
        if len(corpus) == 0:
            raise ValueError("corpus must not be empty")
        for text in corpus:
            if not isinstance(text, str):
                raise TypeError("corpus entries must be strings")
        words = []
        for text in corpus:
            words.extend(_normalise_words(text))
        if not words:
            raise ValueError("corpus must contain at least one word")

        word_freqs = {}
        for word in words:  # first-appearance order is preserved
            word_freqs.setdefault(word, 0)
            word_freqs[word] += 1
        splits = {word: _word_to_symbols(word) for word in word_freqs}

        self.merges = []
        self.vocab = set()
        for symbols in splits.values():
            self.vocab.update(symbols)

        for _ in range(self.num_merges):
            counts = _pair_counts(
                {splits[w]: f for w, f in word_freqs.items()}
            )
            if not counts:
                break
            # max() keeps the first maximal item in insertion order,
            # i.e. the first-seen pair wins ties.
            best_pair, best_count = max(counts.items(), key=lambda kv: kv[1])
            if best_count < self.min_frequency:
                break
            self.merges.append(best_pair)
            merged_symbol = best_pair[0] + best_pair[1]
            self.vocab.add(merged_symbol)
            for word in splits:
                splits[word] = _merge_word(splits[word], best_pair)

        self._fitted = True
        return self

    # -- encoding / decoding ----------------------------------------------
    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("tokenizer has not been fitted yet; call fit() first")

    def encode(self, text):
        """Encode ``text`` into a list of token strings.

        Each word is split into characters plus the end-of-word marker and
        the learned merges are applied greedily in learned order. Words
        never seen in training simply stay as individual characters.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._require_fitted()
        tokens = []
        for word in _normalise_words(text):
            symbols = _word_to_symbols(word)
            for pair in self.merges:
                symbols = _merge_word(symbols, pair)
            tokens.extend(symbols)
        return tokens

    def decode(self, tokens):
        """Reconstruct text from a list of token strings.

        The end-of-word marker is replaced by a space, so the result is
        the normalised (lowercased, alphanumeric-words) form of the text.
        """
        if not isinstance(tokens, (list, tuple)):
            raise TypeError("tokens must be a list of strings")
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError("tokens must be strings")
        self._require_fitted()
        if not tokens:
            return ""
        return "".join(tokens).replace(END_OF_WORD, " ").strip()

    # -- inspection --------------------------------------------------------
    @property
    def vocab_size(self):
        """Number of distinct symbols learned (characters + merges)."""
        return len(self.vocab)

    @property
    def is_fitted(self):
        return self._fitted

    def __repr__(self):
        return (
            f"BPETokenizer(num_merges={self.num_merges}, "
            f"min_frequency={self.min_frequency}, "
            f"merges_learned={len(self.merges)}, "
            f"vocab_size={self.vocab_size})"
        )
