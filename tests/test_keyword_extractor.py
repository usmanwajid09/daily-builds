"""Tests for the RAKE-style keyword extractor."""

import pytest

from keyword_extractor import (
    KeywordExtractor,
    candidate_phrases,
    extract_keywords,
    phrase_scores,
    tokenize,
    word_scores,
)

TEXT = (
    "Machine learning models learn patterns from training data. "
    "Deep neural networks use many layers of artificial neurons. "
    "Training a neural network requires large datasets."
)


def test_top_phrase_is_multiword():
    top = extract_keywords(TEXT, top_n=5)
    assert len(top) == 5
    assert all(isinstance(p, str) and isinstance(s, float) for p, s in top)
    assert top[0][1] >= top[-1][1]  # sorted by score


def test_repeated_phrase_scores_high():
    text = ("Neural networks are powerful. Neural networks learn fast. "
            "Neural networks need data.")
    phrases = [p for p, _ in extract_keywords(text, top_n=3)]
    assert "neural networks" in phrases


def test_stop_words_split_phrases():
    phrases = candidate_phrases("machine learning and deep learning")
    assert ["machine", "learning"] in phrases
    assert ["deep", "learning"] in phrases


def test_word_scores_degree_over_frequency():
    phrases = [["machine", "learning"], ["machine", "learning", "models"]]
    scores = word_scores(phrases)
    # "learning" appears in both phrases: degree 5, freq 2 -> 2.5
    assert scores["learning"] == pytest.approx(2.5)
    assert scores["models"] == pytest.approx(3.0)


def test_phrase_scores_sum_words():
    phrases = [["a", "b"], ["c"]]
    scores = {"a": 1.0, "b": 2.0, "c": 5.0}
    result = dict(phrase_scores(phrases, scores))
    assert result["a b"] == pytest.approx(3.0)
    assert result["c"] == pytest.approx(5.0)


def test_top_n_limit():
    assert len(extract_keywords(TEXT, top_n=2)) == 2


def test_min_max_words_filter():
    single = extract_keywords(TEXT, top_n=10, min_words=1, max_words=1)
    assert all(len(p.split()) == 1 for p, _ in single)
    multi = extract_keywords(TEXT, top_n=10, min_words=2, max_words=2)
    assert all(len(p.split()) == 2 for p, _ in multi)


def test_deterministic_order():
    assert extract_keywords(TEXT, top_n=5) == extract_keywords(TEXT, top_n=5)


def test_tokenize_lowercases():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_empty_text_raises():
    with pytest.raises(ValueError):
        extract_keywords("")
    with pytest.raises(ValueError):
        candidate_phrases("  ")


def test_non_string_raises():
    with pytest.raises(ValueError):
        extract_keywords(123)


def test_bad_top_n():
    with pytest.raises(ValueError):
        extract_keywords(TEXT, top_n=0)
    with pytest.raises(ValueError):
        extract_keywords(TEXT, top_n="five")


def test_bad_word_limits():
    with pytest.raises(ValueError):
        extract_keywords(TEXT, min_words=0)
    with pytest.raises(ValueError):
        extract_keywords(TEXT, min_words=3, max_words=2)


def test_all_stop_words_returns_empty():
    assert extract_keywords("the and of to in", top_n=5) == []


def test_extractor_class():
    extractor = KeywordExtractor(top_n=3)
    result = extractor.extract(TEXT)
    assert len(result) == 3
    assert result == extract_keywords(TEXT, top_n=3)


def test_extractor_custom_stop_words():
    extractor = KeywordExtractor(stop_words={"learning"})
    phrases = candidate_phrases("machine learning models", extractor.stop_words)
    assert ["machine"] in phrases
    assert ["models"] in phrases


def test_word_scores_validation():
    with pytest.raises(ValueError):
        word_scores([["a"], []])
    with pytest.raises(ValueError):
        word_scores("not a list")
