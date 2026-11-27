"""Lexicon-based sentiment analyzer with negation and intensifier handling.

Scores text in [-1, 1]: positive words add weight, negative words subtract
it. A preceding negation ("not", "never", "no") flips the polarity of the
next sentiment word, and intensifiers ("very", "extremely") amplify it.
Per-sentence scores are averaged (weighted by the number of sentiment
words) into a document score, mapped to the labels positive / negative /
neutral.
"""

import re

import numpy as np

POSITIVE = {
    "love": 1.0, "loved": 1.0, "lovely": 0.9, "great": 0.9, "excellent": 1.0,
    "amazing": 1.0, "awesome": 0.9, "fantastic": 1.0, "wonderful": 0.9,
    "good": 0.7, "nice": 0.6, "happy": 0.8, "joy": 0.9, "joyful": 0.9,
    "delight": 0.9, "delighted": 0.9, "pleased": 0.7, "glad": 0.7,
    "best": 1.0, "brilliant": 0.9, "superb": 1.0, "perfect": 1.0,
    "beautiful": 0.8, "enjoy": 0.8, "enjoyed": 0.8, "fun": 0.7,
    "helpful": 0.7, "easy": 0.5, "fast": 0.4, "reliable": 0.7,
    "impressive": 0.8, "outstanding": 1.0, "success": 0.8, "win": 0.7,
    "like": 0.6, "liked": 0.6, "recommend": 0.8, "satisfied": 0.8,
    "comfortable": 0.5, "smooth": 0.5, "clean": 0.4, "bright": 0.4,
    "hopeful": 0.6, "proud": 0.7, "grateful": 0.8, "excited": 0.8,
    "cheap": 0.4, "affordable": 0.5, "valuable": 0.6, "strong": 0.5,
}

NEGATIVE = {
    "hate": -1.0, "hated": -1.0, "terrible": -1.0, "awful": -1.0,
    "horrible": -1.0, "bad": -0.7, "worst": -1.0, "poor": -0.7,
    "sad": -0.8, "angry": -0.9, "annoyed": -0.7, "frustrated": -0.8,
    "disappointed": -0.9, "disappointing": -0.9, "useless": -0.9,
    "boring": -0.7, "slow": -0.5, "broken": -0.9, "buggy": -0.8,
    "expensive": -0.6, "overpriced": -0.8, "fail": -0.8, "failed": -0.8,
    "failure": -0.9, "wrong": -0.7, "problem": -0.6, "issue": -0.5,
    "difficult": -0.5, "hard": -0.4, "confusing": -0.6, "ugly": -0.7,
    "dirty": -0.6, "unreliable": -0.9, "dislike": -0.6, "disliked": -0.6,
    "regret": -0.7, "waste": -0.8, "painful": -0.8, "scary": -0.6,
    "rude": -0.8, "unhelpful": -0.8, "weak": -0.5, "messy": -0.6,
    "annoying": -0.8, "dull": -0.5, "fake": -0.7, "scam": -1.0,
}

LEXICON = {**POSITIVE, **NEGATIVE}

NEGATIONS = {"not", "no", "never", "n't", "cannot", "can't", "won't",
             "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't",
             "couldn't", "shouldn't", "wouldn't", "hardly", "barely"}

INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "really": 1.4, "so": 1.3,
    "absolutely": 1.8, "totally": 1.5, "completely": 1.5, "highly": 1.5,
    "incredibly": 1.8, "super": 1.4, "quite": 1.2, "pretty": 1.2,
    "rather": 1.1, "somewhat": 0.7, "slightly": 0.6, "too": 1.3,
}

# Small words a negation may skip over before reaching a sentiment word.
_SKIP_WORDS = {"a", "an", "the", "is", "was", "are", "were", "it", "this",
               "that", "very", "so", "really"}

_WORD_RE = re.compile(r"[a-z]+n't|[a-z]+(?:'s)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text):
    """Lower-case word tokens (keeps n't contractions as one token)."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return _WORD_RE.findall(text.lower())


def split_sentences(text):
    """Split text into sentences on . ! ? boundaries."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip())]
    return [p for p in parts if p]


def _score_tokens(tokens, lexicon=LEXICON):
    """Score one token list; returns (score, sentiment_word_count)."""
    total = 0.0
    count = 0
    negate_next = False
    multiplier = 1.0
    for token in tokens:
        if token in NEGATIONS:
            negate_next = True
            continue
        if token in INTENSIFIERS:
            multiplier = INTENSIFIERS[token]
            continue
        if token in lexicon:
            weight = lexicon[token] * multiplier
            if negate_next:
                weight = -weight
            total += weight
            count += 1
            negate_next = False
            multiplier = 1.0
        else:
            # A negation only reaches the next sentiment word; it is
            # cancelled once any other content word intervenes.
            if negate_next and token not in _SKIP_WORDS:
                negate_next = False
            multiplier = 1.0
    if count == 0:
        return 0.0, 0
    return float(np.clip(total / count, -1.0, 1.0)), count


def score_sentence(sentence, lexicon=None):
    """Sentiment score of one sentence in [-1, 1]."""
    if not isinstance(sentence, str):
        raise ValueError("sentence must be a string")
    if not sentence.strip():
        raise ValueError("sentence must not be empty")
    score, _ = _score_tokens(tokenize(sentence), lexicon or LEXICON)
    return score


def label(score):
    """Map a score in [-1, 1] to positive / negative / neutral."""
    if not isinstance(score, (int, float)):
        raise ValueError("score must be a number")
    if score != score or score in (float("inf"), float("-inf")):
        raise ValueError("score must be a finite number")
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


def analyze(text, lexicon=None):
    """Analyze a document.

    Returns a dict with the overall ``score`` in [-1, 1], the ``label``,
    the number of ``sentiment_words`` found, and a list of per-sentence
    ``sentences`` dicts with ``text``, ``score`` and ``label``.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    lex = lexicon or LEXICON
    sentences = split_sentences(text)
    per_sentence = []
    weighted_sum = 0.0
    total_words = 0
    for sentence in sentences:
        tokens = tokenize(sentence)
        score, count = _score_tokens(tokens, lex)
        per_sentence.append(
            {"text": sentence, "score": score, "label": label(score)}
        )
        weighted_sum += score * count
        total_words += count
    overall = weighted_sum / total_words if total_words else 0.0
    overall = float(np.clip(overall, -1.0, 1.0))
    return {
        "score": overall,
        "label": label(overall),
        "sentiment_words": total_words,
        "sentences": per_sentence,
    }


class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer (negation + intensifier aware)."""

    def __init__(self, lexicon=None):
        self.lexicon = dict(LEXICON) if lexicon is None else dict(lexicon)
        for word, weight in self.lexicon.items():
            if not isinstance(weight, (int, float)) or not -1.0 <= weight <= 1.0:
                raise ValueError(
                    f"lexicon weight for {word!r} must be a number in [-1, 1]"
                )

    def score(self, text):
        """Overall sentiment score of ``text`` in [-1, 1]."""
        return analyze(text, self.lexicon)["score"]

    def analyze(self, text):
        """Full analysis dict (see :func:`analyze`)."""
        return analyze(text, self.lexicon)
