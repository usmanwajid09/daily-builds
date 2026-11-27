"""Readability scorer: Flesch Reading Ease, Flesch-Kincaid Grade Level,
Gunning Fog Index and SMOG (approximate), plus raw counts.

Syllables are estimated with a vowel-group heuristic (with a silent-e
rule), which is the standard approximation used by pure-Python
readability tools.
"""

import re

import numpy as np

_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

# Words whose syllable count the heuristic gets wrong; handled explicitly.
_SYLLABLE_EXCEPTIONS = {
    "the": 1, "a": 1, "an": 1, "i": 1, "you": 1, "he": 1, "she": 1,
    "we": 1, "they": 1, "are": 1, "were": 1, "been": 1, "one": 1,
    "two": 1, "three": 1, "four": 1, "five": 1, "seven": 2, "eight": 1,
    "nine": 1, "ten": 1, "people": 2, "business": 2, "interesting": 4,
    "every": 2, "different": 3, "family": 3, "really": 2, "probably": 3,
    "beautiful": 3, "education": 4, "government": 3, "important": 3,
    "area": 2, "idea": 2, "science": 2, "fire": 1, "hour": 1,
}


def count_syllables(word):
    """Estimate the syllable count of one word (vowel-group heuristic)."""
    if not isinstance(word, str):
        raise ValueError("word must be a string")
    w = word.strip().lower()
    if not w:
        raise ValueError("word must not be empty")
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        raise ValueError("word must contain letters")
    if w in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[w]
    if len(w) <= 3:
        return 1
    count = len(_VOWEL_GROUP_RE.findall(w))
    if w.endswith("e"):
        count -= 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy":
        count += 1
    if w.endswith("ed") and len(w) > 2 and w[-3] not in "aeiouy":
        count -= 1
    return max(1, count)


def split_sentences(text):
    """Split text into sentences on . ! ? boundaries."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip())]
    return [p for p in parts if p]


def tokenize(text):
    """Word tokens (letters only, apostrophes kept inside words)."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return _WORD_RE.findall(text)


def count_stats(text):
    """Return counts: sentences, words, syllables, complex words (>=3 syllables)."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    sentences = split_sentences(text)
    words = tokenize(text)
    if not words:
        raise ValueError("text must contain at least one word")
    syllables = int(np.sum([count_syllables(w) for w in words]))
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    return {
        "sentences": len(sentences),
        "words": len(words),
        "syllables": syllables,
        "complex_words": complex_words,
        "avg_words_per_sentence": len(words) / len(sentences),
        "avg_syllables_per_word": syllables / len(words),
    }


def flesch_reading_ease(text):
    """Flesch Reading Ease: 0-100, higher = easier to read."""
    stats = count_stats(text)
    score = (206.835 - 1.015 * stats["avg_words_per_sentence"]
             - 84.6 * stats["avg_syllables_per_word"])
    return float(np.clip(score, 0.0, 100.0))


def flesch_kincaid_grade(text):
    """Flesch-Kincaid Grade Level (US school grade)."""
    stats = count_stats(text)
    grade = (0.39 * stats["avg_words_per_sentence"]
             + 11.8 * stats["avg_syllables_per_word"] - 15.59)
    return float(max(0.0, grade))


def gunning_fog(text):
    """Gunning Fog Index (years of education needed)."""
    stats = count_stats(text)
    if stats["words"] == 0:
        raise ValueError("text must contain at least one word")
    pct_complex = 100.0 * stats["complex_words"] / stats["words"]
    return float(0.4 * (stats["avg_words_per_sentence"] + pct_complex))


def smog_index(text):
    """SMOG grade (approximate; standard formula needs >= 30 sentences,
    here it is applied to any text with at least 3 sentences)."""
    stats = count_stats(text)
    if stats["sentences"] < 3:
        raise ValueError("SMOG needs at least 3 sentences")
    return float(1.0430 * np.sqrt(stats["complex_words"] * 30.0 / stats["sentences"]) + 3.1291)


def ease_label(score):
    """Plain-English label for a Flesch Reading Ease score."""
    if not isinstance(score, (int, float)):
        raise ValueError("score must be a number")
    if score >= 90:
        return "Very Easy"
    if score >= 80:
        return "Easy"
    if score >= 70:
        return "Fairly Easy"
    if score >= 60:
        return "Standard"
    if score >= 50:
        return "Fairly Difficult"
    if score >= 30:
        return "Difficult"
    return "Very Difficult"


def score_all(text):
    """Compute every readability metric for ``text`` in one dict."""
    stats = count_stats(text)
    result = {
        "sentences": stats["sentences"],
        "words": stats["words"],
        "syllables": stats["syllables"],
        "complex_words": stats["complex_words"],
        "avg_words_per_sentence": round(stats["avg_words_per_sentence"], 2),
        "avg_syllables_per_word": round(stats["avg_syllables_per_word"], 2),
        "flesch_reading_ease": round(flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(flesch_kincaid_grade(text), 2),
        "gunning_fog": round(gunning_fog(text), 2),
    }
    result["ease_label"] = ease_label(result["flesch_reading_ease"])
    try:
        result["smog"] = round(smog_index(text), 2)
    except ValueError:
        result["smog"] = None
    return result


class ReadabilityScorer:
    """Readability metrics for a text (see :func:`score_all`)."""

    def score(self, text):
        """All readability metrics for ``text`` as a dict."""
        return score_all(text)
