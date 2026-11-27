"""RAKE-style keyword/keyphrase extractor."""

from keyword_extractor.model import (
    STOP_WORDS,
    KeywordExtractor,
    candidate_phrases,
    extract_keywords,
    phrase_scores,
    split_sentences,
    tokenize,
    word_scores,
)

__all__ = [
    "STOP_WORDS",
    "KeywordExtractor",
    "candidate_phrases",
    "extract_keywords",
    "phrase_scores",
    "split_sentences",
    "tokenize",
    "word_scores",
]
