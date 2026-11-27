"""Plagiarism / text-overlap detector.

Compares two texts with word n-gram shingles: Jaccard similarity and
containment (how much of the shorter text is inside the other), plus a
longest-common-subsequence ratio over word tokens. The report lists the
most overlapping shingles so a human can see exactly what matches.
"""

import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lower-case word tokens."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return _WORD_RE.findall(text.lower())


def shingles(text, n=3):
    """Set of word n-gram shingles (``n`` consecutive words, joined)."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    words = tokenize(text)
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(text_a, text_b, n=3):
    """Jaccard similarity of the two texts' shingle sets in [0, 1]."""
    set_a = shingles(text_a, n)
    set_b = shingles(text_b, n)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def containment(text_a, text_b, n=3):
    """Fraction of the smaller shingle set contained in the larger one."""
    set_a = shingles(text_a, n)
    set_b = shingles(text_b, n)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def lcs_ratio(text_a, text_b):
    """Longest-common-subsequence ratio over word tokens.

    LCS length divided by the length of the shorter token list, in [0, 1].
    """
    a = tokenize(text_a)
    b = tokenize(text_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Dynamic programming over the shorter list for memory efficiency.
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1] / min(len(tokenize(text_a)), len(tokenize(text_b)))


def overlap_shingles(text_a, text_b, n=3):
    """Shingles shared by both texts, most frequent first."""
    set_a = shingles(text_a, n)
    set_b = shingles(text_b, n)
    shared = set_a & set_b
    counts = Counter()
    for shingle in shared:
        counts[shingle] = tokenize(text_a).count(shingle.split()[0]) \
            + tokenize(text_b).count(shingle.split()[0])
    return sorted(shared, key=lambda s: (-counts[s], s))


def verdict(score):
    """Plain-English verdict for a similarity score in [0, 1]."""
    if not isinstance(score, (int, float)):
        raise ValueError("score must be a number")
    if score != score or score < 0.0 or score > 1.0:
        raise ValueError("score must be a number in [0, 1]")
    if score >= 0.5:
        return "high overlap — likely copied"
    if score >= 0.2:
        return "moderate overlap — possible paraphrase"
    if score >= 0.05:
        return "low overlap — mostly original"
    return "no meaningful overlap"


def compare(text_a, text_b, n=3):
    """Compare two texts; returns a per-pair report dict.

    Keys: ``jaccard``, ``containment``, ``lcs_ratio``,
    ``combined_score`` (mean of the three), ``verdict``,
    ``shared_shingles`` (count) and ``top_shingles`` (up to 10).
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        raise ValueError("both texts must be strings")
    if not text_a.strip() or not text_b.strip():
        raise ValueError("both texts must not be empty")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    jac = jaccard(text_a, text_b, n)
    cont = containment(text_a, text_b, n)
    lcs = lcs_ratio(text_a, text_b)
    combined = float(np.mean([jac, cont, lcs]))
    shared = overlap_shingles(text_a, text_b, n)
    return {
        "jaccard": round(jac, 4),
        "containment": round(cont, 4),
        "lcs_ratio": round(lcs, 4),
        "combined_score": round(combined, 4),
        "verdict": verdict(combined),
        "shared_shingles": len(shared),
        "top_shingles": shared[:10],
        "n_gram": n,
    }


def compare_many(texts, n=3):
    """Compare every pair in a list of texts; returns a list of reports.

    Each report is the :func:`compare` dict plus ``i`` and ``j`` indices.
    """
    if not isinstance(texts, list) or len(texts) < 2:
        raise ValueError("texts must be a list of at least 2 strings")
    results = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            report = compare(texts[i], texts[j], n)
            report["i"] = i
            report["j"] = j
            results.append(report)
    return results


class PlagiarismDetector:
    """Word n-gram shingle overlap detector (see :func:`compare`)."""

    def __init__(self, n=3):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self.n = n

    def compare(self, text_a, text_b):
        """Per-pair report for two texts."""
        return compare(text_a, text_b, self.n)

    def compare_many(self, texts):
        """Pairwise reports for a list of texts."""
        return compare_many(texts, self.n)
