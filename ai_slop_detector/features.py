"""Stylometric feature extraction for the AI-slop detector.

These are hand-picked, human-interpretable "stylometric" signals -- word
choice, punctuation, sentence-length rhythm, contraction use, etc. This is
NOT semantic understanding and it is NOT perplexity-under-a-real-LM. It is a
small, from-scratch, educational feature set that captures some of the more
obvious surface tells of typical LLM prose (hedge phrases, uniform sentence
length, corporate-blog vocabulary) versus typical informal human writing
(contractions, slang, uneven sentence rhythm, personal anecdotes). See the
README "Limitations" section for what this deliberately does not attempt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CONTRACTION_RE = re.compile(r"\b[a-zA-Z]+'(?:t|re|ve|ll|d|m|s)\b", re.IGNORECASE)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

# Phrases/words that show up disproportionately often in default-style LLM
# output ("AI tells"). Checked as lowercased substrings so multi-word
# phrases work too.
AI_MARKER_PHRASES = [
    "moreover", "furthermore", "additionally", "in conclusion",
    "it is important to note", "it's important to note", "delve",
    "tapestry", "boast", "boasts", "navigate the", "landscape", "realm",
    "notably", "robust", "seamless", "leverage", "foster", "holistic",
    "comprehensive", "underscore", "embark", "unlock", "elevate",
    "in today's fast-paced world", "in the world of", "when it comes to",
    "ultimately", "overall,", "plays a crucial role", "plays a vital role",
    "it is worth noting", "in summary", "to sum up", "a testament to",
    "unparalleled", "cutting-edge", "game-changer", "game changer",
]

FILLER_SLANG_WORDS = {
    "lol", "tbh", "idk", "gonna", "wanna", "kinda", "sorta", "ngl", "omg",
    "haha", "yeah", "gotta", "dunno", "ain't", "y'know",
}

FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "we", "us", "our", "ours"}


@dataclass
class FeatureVector:
    """Named stylometric features for a single piece of text.

    Every field is a plain float. `FEATURE_NAMES` gives the column order,
    which must match `to_array`.
    """

    avg_sentence_len_words: float
    sentence_len_stdev: float
    avg_word_len_chars: float
    type_token_ratio: float
    contraction_ratio_per_100w: float
    ai_marker_ratio_per_100w: float
    first_person_ratio_per_100w: float
    exclaim_question_ratio: float
    avg_syllables_per_word: float
    comma_semicolon_ratio_per_sentence: float
    filler_slang_ratio_per_100w: float

    def to_array(self):
        import numpy as np

        return np.array([getattr(self, name) for name in FEATURE_NAMES], dtype=float)


FEATURE_NAMES = [
    "avg_sentence_len_words",
    "sentence_len_stdev",
    "avg_word_len_chars",
    "type_token_ratio",
    "contraction_ratio_per_100w",
    "ai_marker_ratio_per_100w",
    "first_person_ratio_per_100w",
    "exclaim_question_ratio",
    "avg_syllables_per_word",
    "comma_semicolon_ratio_per_sentence",
    "filler_slang_ratio_per_100w",
]


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts


def _count_syllables(word: str) -> int:
    groups = _VOWEL_GROUP_RE.findall(word)
    count = len(groups)
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def extract_features(text: str) -> FeatureVector:
    """Extract the stylometric feature vector for `text`.

    Raises ValueError on empty/whitespace-only or word-less input rather
    than silently returning a zero-vector, since a silent zero-vector would
    be indistinguishable from a genuinely feature-poor sample.
    """
    if not text or not text.strip():
        raise ValueError("extract_features() requires non-empty text")

    sentences = _split_sentences(text)
    words = _WORD_RE.findall(text)
    n_words = len(words)

    if n_words == 0:
        raise ValueError("extract_features() requires text containing at least one word")

    n_sentences = max(len(sentences), 1)
    sentence_lengths = [len(_WORD_RE.findall(s)) for s in sentences] or [n_words]

    avg_sentence_len = sum(sentence_lengths) / len(sentence_lengths)
    if len(sentence_lengths) > 1:
        mean = avg_sentence_len
        variance = sum((x - mean) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        sentence_len_stdev = variance ** 0.5
    else:
        sentence_len_stdev = 0.0

    avg_word_len = sum(len(w) for w in words) / n_words

    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    type_token_ratio = len(unique_words) / n_words

    n_contractions = len(_CONTRACTION_RE.findall(text))
    contraction_ratio = n_contractions / n_words * 100

    lower_text = text.lower()
    n_ai_markers = sum(lower_text.count(phrase) for phrase in AI_MARKER_PHRASES)
    ai_marker_ratio = n_ai_markers / n_words * 100

    n_first_person = sum(1 for w in lower_words if w in FIRST_PERSON_WORDS)
    first_person_ratio = n_first_person / n_words * 100

    n_exclaim_question = text.count("!") + text.count("?")
    exclaim_question_ratio = n_exclaim_question / n_sentences

    total_syllables = sum(_count_syllables(w) for w in words)
    avg_syllables = total_syllables / n_words

    n_comma_semicolon = text.count(",") + text.count(";")
    comma_semicolon_ratio = n_comma_semicolon / n_sentences

    n_filler = sum(1 for w in lower_words if w in FILLER_SLANG_WORDS)
    filler_ratio = n_filler / n_words * 100

    return FeatureVector(
        avg_sentence_len_words=avg_sentence_len,
        sentence_len_stdev=sentence_len_stdev,
        avg_word_len_chars=avg_word_len,
        type_token_ratio=type_token_ratio,
        contraction_ratio_per_100w=contraction_ratio,
        ai_marker_ratio_per_100w=ai_marker_ratio,
        first_person_ratio_per_100w=first_person_ratio,
        exclaim_question_ratio=exclaim_question_ratio,
        avg_syllables_per_word=avg_syllables,
        comma_semicolon_ratio_per_sentence=comma_semicolon_ratio,
        filler_slang_ratio_per_100w=filler_ratio,
    )
