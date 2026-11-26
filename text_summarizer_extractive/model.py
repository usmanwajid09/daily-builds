"""Extractive text summarizer, from scratch.

Sentences are ranked by term frequency: content words (stopwords
removed) are scored by how often they appear in the document, each
sentence's score is the average word score of its content words, and
the top-N sentences are returned in their original order. Only numpy
(and the standard library) is used.
"""

import re

import numpy as np

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z]+")

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about
    into through after over between out against during without before
    under around among to from in on as is am are was were be been being
    have has had do does did will would shall should can could may might
    must i you he she it we they me him her us them my your his its our
    their this that these those there here what which who whom how why
    not no yes so very just also than too then once here there all any
    both each few more most other some such only own same s t don should
    now
    """.split()
)


def split_sentences(text):
    """Split text into sentences on '.', '!' and '?' terminators.

    Returns a list of stripped, non-empty sentence strings.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text):
    """Lowercase a string and return its alphabetic word tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _WORD_RE.findall(text.lower())


def content_words(text):
    """Word tokens of a string with stopwords removed."""
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]


def word_frequencies(sentences):
    """Term-frequency dict of content words across the sentences."""
    if not isinstance(sentences, (list, tuple)):
        raise TypeError("sentences must be a list of strings")
    counts = {}
    total = 0
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("each sentence must be a string")
        for word in content_words(sentence):
            counts[word] = counts.get(word, 0) + 1
            total += 1
    if total == 0:
        raise ValueError("no content words found (text is only stopwords)")
    return {word: count / total for word, count in counts.items()}


def score_sentences(sentences, frequencies=None):
    """Average content-word TF score for each sentence.

    Returns a numpy array of scores, one per sentence. A sentence with
    no content words scores 0.0.
    """
    if not isinstance(sentences, (list, tuple)) or len(sentences) == 0:
        raise ValueError("at least one sentence is required")
    if frequencies is None:
        frequencies = word_frequencies(sentences)
    scores = np.zeros(len(sentences), dtype=float)
    for i, sentence in enumerate(sentences):
        words = content_words(sentence)
        if words:
            scores[i] = float(
                np.mean([frequencies.get(w, 0.0) for w in words])
            )
    return scores


def summarize(text, n_sentences=2):
    """Return the top-n sentences of ``text`` in their original order.

    Parameters
    ----------
    text : str
        The document to summarize. Must contain at least one sentence.
    n_sentences : int
        How many sentences to keep. Must be >= 1 and no larger than
        the number of sentences in the text.

    Returns
    -------
    list of str
        The selected sentences, in the order they appear in ``text``.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    if isinstance(n_sentences, bool) or not isinstance(n_sentences, int):
        raise TypeError("n_sentences must be an integer")
    if n_sentences < 1:
        raise ValueError("n_sentences must be at least 1")
    sentences = split_sentences(text)
    if len(sentences) == 0:
        raise ValueError("text contains no sentences")
    if len(sentences) == 1:
        return sentences
    if n_sentences > len(sentences):
        raise ValueError(
            f"n_sentences ({n_sentences}) exceeds the number of sentences "
            f"in the text ({len(sentences)})"
        )
    scores = score_sentences(sentences)
    ranked = np.argsort(-scores, kind="stable")[:n_sentences]
    chosen = sorted(int(i) for i in ranked)
    return [sentences[i] for i in chosen]


class ExtractiveSummarizer:
    """Extractive summarizer with a fixed summary length.

    Parameters
    ----------
    n_sentences : int
        Default number of sentences per summary. Must be >= 1.
    """

    def __init__(self, n_sentences=2):
        if isinstance(n_sentences, bool) or not isinstance(n_sentences, int):
            raise TypeError("n_sentences must be an integer")
        if n_sentences < 1:
            raise ValueError("n_sentences must be at least 1")
        self.n_sentences = n_sentences

    def summarize(self, text, n_sentences=None):
        """Summarize ``text`` to ``n_sentences`` sentences (default: the
        value set at construction). Returns the selected sentences in
        their original order."""
        n = self.n_sentences if n_sentences is None else n_sentences
        return summarize(text, n_sentences=n)

    def scores(self, text):
        """Sentence scores for ``text``, in original order."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")
        return score_sentences(split_sentences(text))
