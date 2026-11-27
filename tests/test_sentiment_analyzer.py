"""Tests for the sentiment analyzer."""

import pytest

from sentiment_analyzer import (
    SentimentAnalyzer,
    analyze,
    label,
    score_sentence,
    split_sentences,
    tokenize,
)


def test_positive_text():
    result = analyze("I love this amazing product. It is wonderful!")
    assert result["label"] == "positive"
    assert result["score"] > 0.5


def test_negative_text():
    result = analyze("I hate this terrible product. It is awful and broken.")
    assert result["label"] == "negative"
    assert result["score"] < -0.5


def test_neutral_text_no_sentiment_words():
    result = analyze("The package arrived on Tuesday.")
    assert result["score"] == 0.0
    assert result["label"] == "neutral"
    assert result["sentiment_words"] == 0


def test_negation_flips_polarity():
    assert score_sentence("This is not good.") < 0
    assert score_sentence("This is not bad.") > 0


def test_negation_never():
    assert score_sentence("I never liked this.") < 0


def test_intensifier_amplifies():
    plain = score_sentence("This is good.")
    strong = score_sentence("This is very good.")
    assert strong > plain > 0


def test_intensifier_extremely():
    assert score_sentence("This is extremely terrible.") < -0.9


def test_per_sentence_breakdown():
    result = analyze("I love it. I hate the price.")
    assert len(result["sentences"]) == 2
    assert result["sentences"][0]["label"] == "positive"
    assert result["sentences"][1]["label"] == "negative"


def test_score_bounds():
    for text in ["I love love love it!", "I hate hate hate it!", "Meh."]:
        score = analyze(text)["score"]
        assert -1.0 <= score <= 1.0


def test_label_thresholds():
    assert label(0.5) == "positive"
    assert label(-0.5) == "negative"
    assert label(0.05) == "neutral"
    assert label(0.0) == "neutral"
    assert label(0.1) == "neutral"
    assert label(0.11) == "positive"


def test_tokenize_keeps_contractions():
    assert "don't" in tokenize("I don't like it")
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_empty_text_raises():
    with pytest.raises(ValueError):
        analyze("")
    with pytest.raises(ValueError):
        analyze("   ")


def test_non_string_raises():
    with pytest.raises(ValueError):
        analyze(42)
    with pytest.raises(ValueError):
        score_sentence(None)


def test_empty_sentence_raises():
    with pytest.raises(ValueError):
        score_sentence("")


def test_label_validation():
    with pytest.raises(ValueError):
        label("positive")
    with pytest.raises(ValueError):
        label(float("nan"))


def test_analyzer_class_matches_function():
    analyzer = SentimentAnalyzer()
    assert analyzer.score("I love it") == analyze("I love it")["score"]
    assert analyzer.analyze("I hate it")["label"] == "negative"


def test_custom_lexicon():
    analyzer = SentimentAnalyzer(lexicon={"brilliant": 1.0})
    assert analyzer.score("This is brilliant") == 1.0
    assert analyzer.score("This is good") == 0.0


def test_custom_lexicon_bad_weight():
    with pytest.raises(ValueError):
        SentimentAnalyzer(lexicon={"good": 5.0})


def test_mixed_review_near_neutral():
    result = analyze("I love the design but I hate the price.")
    assert -0.3 < result["score"] < 0.3
