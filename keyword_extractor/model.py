"""RAKE-style keyword/keyphrase extractor.

Splits text on stop words into candidate phrases, scores each word by
degree / frequency (how often it co-occurs with other words inside a
phrase), and ranks phrases by the sum of their member-word scores.
Pure Python + numpy; no external NLP libraries.
"""

import re
from collections import Counter

import numpy as np

STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "while", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "don", "should", "now", "is", "am", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "shall", "could", "ought",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "as",
    "of", "without", "around", "among", "also", "may", "might", "must",
    "per", "via", "upon", "within", "across", "like",
})

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(text):
    """Lower-case word tokens (keeps hyphenated words together)."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return _WORD_RE.findall(text.lower())


def split_sentences(text):
    """Split text into sentences."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip())]
    return [p for p in parts if p]


def candidate_phrases(text, stop_words=None):
    """Split text into candidate phrases at stop words and punctuation.

    Returns a list of word lists (each list is one candidate phrase).
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    stops = STOP_WORDS if stop_words is None else set(stop_words)
    phrases = []
    for sentence in split_sentences(text):
        current = []
        for word in tokenize(sentence):
            if word in stops:
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append(word)
        if current:
            phrases.append(current)
    return phrases


def word_scores(phrases):
    """RAKE word scores: degree(word) / frequency(word).

    Degree counts how often a word co-occurs with other words in the
    same phrase (a word in a phrase of length n contributes n to its
    degree); frequency is its phrase count.
    """
    if not isinstance(phrases, list):
        raise ValueError("phrases must be a list of word lists")
    freq = Counter()
    degree = Counter()
    for phrase in phrases:
        if not isinstance(phrase, list) or not phrase:
            raise ValueError("each phrase must be a non-empty list of words")
        for word in phrase:
            freq[word] += 1
            degree[word] += len(phrase)
    return {word: degree[word] / freq[word] for word in freq}


def phrase_scores(phrases, scores=None):
    """Score each candidate phrase as the sum of its word scores."""
    if not isinstance(phrases, list):
        raise ValueError("phrases must be a list of word lists")
    if scores is None:
        scores = word_scores(phrases)
    return [(" ".join(phrase),
             float(np.sum([scores.get(w, 0.0) for w in phrase])))
            for phrase in phrases]


def extract_keywords(text, top_n=10, min_words=1, max_words=4,
                     stop_words=None):
    """Extract the top-N keyphrases from ``text``.

    Returns a list of ``(phrase, score)`` tuples sorted by score
    (highest first), with ties broken alphabetically for determinism.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if not isinstance(min_words, int) or min_words < 1:
        raise ValueError("min_words must be a positive integer")
    if not isinstance(max_words, int) or max_words < min_words:
        raise ValueError("max_words must be an integer >= min_words")
    phrases = candidate_phrases(text, stop_words)
    if not phrases:
        return []
    # Score words on the original (unsplit) phrases first, so long
    # phrases keep their true RAKE degree/frequency scores.
    scores = word_scores(phrases)
    # Phrases longer than max_words are split into sliding windows of
    # max_words words so long noun phrases still contribute keyphrases.
    expanded = []
    for phrase in phrases:
        if len(phrase) <= max_words:
            expanded.append(phrase)
        else:
            for i in range(len(phrase) - max_words + 1):
                expanded.append(phrase[i:i + max_words])
    phrases = expanded
    scored = phrase_scores(phrases, scores)
    # De-duplicate identical phrases, keeping the highest score.
    best = {}
    for phrase, score in scored:
        words = phrase.split()
        if min_words <= len(words) <= max_words:
            best[phrase] = max(best.get(phrase, 0.0), score)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_n]


class KeywordExtractor:
    """RAKE-style keyphrase extractor with a configurable stop-word set."""

    def __init__(self, stop_words=None, top_n=10):
        self.stop_words = set(STOP_WORDS) if stop_words is None else set(stop_words)
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
        self.top_n = top_n

    def extract(self, text, top_n=None):
        """Return the top-N ``(phrase, score)`` keyphrases of ``text``."""
        n = self.top_n if top_n is None else top_n
        if not isinstance(n, int) or n < 1:
            raise ValueError("top_n must be a positive integer")
        return extract_keywords(text, top_n=n, stop_words=self.stop_words)
