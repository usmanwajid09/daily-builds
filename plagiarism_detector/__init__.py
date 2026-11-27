"""Plagiarism / text-overlap detector based on word n-gram shingles."""

from plagiarism_detector.model import (
    PlagiarismDetector,
    compare,
    compare_many,
    containment,
    jaccard,
    lcs_ratio,
    overlap_shingles,
    shingles,
    tokenize,
    verdict,
)

__all__ = [
    "PlagiarismDetector",
    "compare",
    "compare_many",
    "containment",
    "jaccard",
    "lcs_ratio",
    "overlap_shingles",
    "shingles",
    "tokenize",
    "verdict",
]
