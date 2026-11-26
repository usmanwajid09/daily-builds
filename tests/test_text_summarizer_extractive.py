"""Tests for the extractive text summarizer."""

import numpy as np
import pytest

from text_summarizer_extractive import (
    ExtractiveSummarizer,
    content_words,
    score_sentences,
    split_sentences,
    summarize,
    tokenize,
    word_frequencies,
)
from text_summarizer_extractive.demo import KEY_TERMS, PARAGRAPH

TEXT = (
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy. "
    "Plants capture sunlight with chlorophyll in their leaves. "
    "During photosynthesis, plants absorb carbon dioxide from the air and water from the soil."
)


# --- split_sentences ---

def test_split_on_period_question_and_exclamation():
    text = "First sentence. Second one! Third one? Fourth."
    assert split_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Third one?",
        "Fourth.",
    ]


def test_split_strips_whitespace_and_drops_empties():
    assert split_sentences("  Hello world.   Goodbye world.  ") == [
        "Hello world.",
        "Goodbye world.",
    ]


def test_split_non_string_raises_type_error():
    with pytest.raises(TypeError, match="must be a string"):
        split_sentences(123)


def test_split_single_sentence_unchanged():
    assert split_sentences("Only one sentence here.") == ["Only one sentence here."]


# --- tokenize / content_words / word_frequencies ---

def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, WORLD! 42 times.") == ["hello", "world", "times"]


def test_content_words_removes_stopwords():
    words = content_words("The quick brown fox jumps over the lazy dog")
    assert "the" not in words and "over" not in words
    assert words == ["quick", "brown", "fox", "jumps", "lazy", "dog"]


def test_word_frequencies_sum_to_one():
    freqs = word_frequencies(split_sentences(TEXT))
    assert abs(sum(freqs.values()) - 1.0) < 1e-9
    assert freqs["photosynthesis"] > freqs["chlorophyll"]


def test_word_frequencies_only_stopwords_raises():
    with pytest.raises(ValueError, match="no content words"):
        word_frequencies(["The and of the.", "It is a."])


def test_word_frequencies_non_string_sentence_raises():
    with pytest.raises(TypeError, match="must be a string"):
        word_frequencies(["Fine sentence.", 42])


# --- score_sentences ---

def test_score_sentences_returns_array_in_original_order():
    scores = score_sentences(split_sentences(TEXT))
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)
    assert np.all(scores >= 0)


def test_score_repeated_key_words_rank_highest():
    # The sentence repeating the document's most frequent words wins.
    text = "Cats sleep all day. Cats love cats and cats love sleep. Dogs bark."
    scores = score_sentences(split_sentences(text))
    assert int(np.argmax(scores)) == 1


def test_score_empty_list_raises():
    with pytest.raises(ValueError, match="at least one sentence"):
        score_sentences([])


# --- summarize validation ---

def test_summarize_empty_text_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize("")


def test_summarize_whitespace_only_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize("   \n\t  ")


def test_summarize_non_string_raises_type_error():
    with pytest.raises(TypeError, match="must be a string"):
        summarize(None)


def test_summarize_n_zero_raises():
    with pytest.raises(ValueError, match="at least 1"):
        summarize(TEXT, n_sentences=0)


def test_summarize_n_too_large_raises():
    with pytest.raises(ValueError, match="exceeds the number of sentences"):
        summarize(TEXT, n_sentences=4)


def test_summarize_n_non_integer_raises():
    with pytest.raises(TypeError, match="must be an integer"):
        summarize(TEXT, n_sentences=1.5)
    with pytest.raises(TypeError, match="must be an integer"):
        summarize(TEXT, n_sentences=True)


# --- summarize behaviour ---

def test_summarize_single_sentence_returns_it():
    assert summarize("Just one sentence here.") == ["Just one sentence here."]


def test_summarize_returns_top_n_in_original_order():
    summary = summarize(TEXT, n_sentences=2)
    assert len(summary) == 2
    sentences = split_sentences(TEXT)
    assert all(s in sentences for s in summary)
    # Original order preserved: indices strictly increasing.
    indices = [sentences.index(s) for s in summary]
    assert indices == sorted(indices)


def test_summarize_all_sentences_returns_everything():
    assert summarize(TEXT, n_sentences=3) == split_sentences(TEXT)


def test_summarize_prefers_repeated_key_words():
    text = (
        "Quantum computers use qubits for calculation. "
        "The weather today is sunny and warm. "
        "Quantum computers promise faster quantum algorithms for quantum problems."
    )
    summary = summarize(text, n_sentences=1)
    assert summary == [
        "Quantum computers promise faster quantum algorithms for quantum problems."
    ]


# --- class API ---

def test_summarizer_class_matches_function():
    clf = ExtractiveSummarizer(n_sentences=2)
    assert clf.summarize(TEXT) == summarize(TEXT, n_sentences=2)
    assert clf.summarize(TEXT, n_sentences=1) == summarize(TEXT, n_sentences=1)


def test_summarizer_class_invalid_n_raises():
    with pytest.raises(ValueError, match="at least 1"):
        ExtractiveSummarizer(n_sentences=0)
    with pytest.raises(TypeError, match="must be an integer"):
        ExtractiveSummarizer(n_sentences="two")


def test_summarizer_scores_match_function():
    clf = ExtractiveSummarizer()
    scores = clf.scores(TEXT)
    assert np.allclose(scores, score_sentences(split_sentences(TEXT)))


# --- demo behaviour ---

def test_demo_paragraph_summary_contains_key_terms():
    summary = summarize(PARAGRAPH, n_sentences=2)
    assert len(summary) == 2
    joined = " ".join(summary).lower()
    for term in KEY_TERMS:
        assert term in joined, f"key term {term!r} missing from summary"
    assert len(split_sentences(PARAGRAPH)) == 8
