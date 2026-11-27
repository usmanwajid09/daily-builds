"""Lexicon-based sentiment analyzer with negation and intensifier handling."""

from sentiment_analyzer.model import (
    INTENSIFIERS,
    LEXICON,
    NEGATIONS,
    NEGATIVE,
    POSITIVE,
    SentimentAnalyzer,
    analyze,
    label,
    score_sentence,
    split_sentences,
    tokenize,
)

__all__ = [
    "INTENSIFIERS",
    "LEXICON",
    "NEGATIONS",
    "NEGATIVE",
    "POSITIVE",
    "SentimentAnalyzer",
    "analyze",
    "label",
    "score_sentence",
    "split_sentences",
    "tokenize",
]
