"""Tests for the readability scorer."""

import pytest

from readability_scorer import (
    ReadabilityScorer,
    count_stats,
    count_syllables,
    ease_label,
    flesch_kincaid_grade,
    flesch_reading_ease,
    gunning_fog,
    score_all,
    smog_index,
    tokenize,
)

SIMPLE = "The cat sat on the mat. It was a warm day. The sun was bright."
COMPLEX = (
    "The implementation of comprehensive regulatory frameworks "
    "necessitates the consideration of multifaceted socioeconomic "
    "variables. Consequently, policymakers must evaluate the "
    "interdependencies between institutional structures and "
    "macroeconomic indicators. Furthermore, the quantification of "
    "long-term consequences requires sophisticated analytical "
    "methodologies and extensive empirical investigation."
)


def test_simple_text_is_easy():
    assert flesch_reading_ease(SIMPLE) > 90
    assert ease_label(flesch_reading_ease(SIMPLE)) == "Very Easy"


def test_complex_text_is_hard():
    ease = flesch_reading_ease(COMPLEX)
    assert ease < 30
    assert ease_label(ease) == "Very Difficult"
    assert flesch_kincaid_grade(COMPLEX) > 12


def test_simple_grade_lower_than_complex():
    assert flesch_kincaid_grade(SIMPLE) < flesch_kincaid_grade(COMPLEX)


def test_gunning_fog_ordering():
    assert gunning_fog(SIMPLE) < gunning_fog(COMPLEX)


def test_smog_needs_three_sentences():
    with pytest.raises(ValueError):
        smog_index("One sentence only. Two sentences here.")
    assert smog_index(COMPLEX) > 0


def test_smog_none_for_short_text():
    result = score_all("One short sentence. Two short sentences.")
    assert result["smog"] is None


def test_count_stats():
    stats = count_stats("The cat sat. The dog ran.")
    assert stats["sentences"] == 2
    assert stats["words"] == 6
    assert stats["syllables"] == 6
    assert stats["complex_words"] == 0


def test_count_syllables_known_words():
    assert count_syllables("cat") == 1
    assert count_syllables("beautiful") == 3
    assert count_syllables("education") == 4
    assert count_syllables("reading") == 2


def test_count_syllables_silent_e():
    assert count_syllables("cake") == 1
    assert count_syllables("make") == 1


def test_score_all_keys():
    result = score_all(SIMPLE)
    for key in ("sentences", "words", "syllables", "complex_words",
                "flesch_reading_ease", "flesch_kincaid_grade",
                "gunning_fog", "smog", "ease_label"):
        assert key in result


def test_ease_label_bands():
    assert ease_label(95) == "Very Easy"
    assert ease_label(85) == "Easy"
    assert ease_label(75) == "Fairly Easy"
    assert ease_label(65) == "Standard"
    assert ease_label(55) == "Fairly Difficult"
    assert ease_label(40) == "Difficult"
    assert ease_label(10) == "Very Difficult"


def test_ease_label_validation():
    with pytest.raises(ValueError):
        ease_label("easy")


def test_flesch_ease_clipped():
    score = flesch_reading_ease("I am Sam. I am Sam. I am Sam. I am Sam.")
    assert 0.0 <= score <= 100.0


def test_tokenize():
    assert tokenize("Hello, world!") == ["Hello", "world"]


def test_empty_text_raises():
    with pytest.raises(ValueError):
        score_all("")
    with pytest.raises(ValueError):
        count_stats("   ")


def test_non_string_raises():
    with pytest.raises(ValueError):
        score_all(42)
    with pytest.raises(ValueError):
        count_syllables(7)


def test_no_words_raises():
    with pytest.raises(ValueError):
        count_stats("!!! ???")


def test_scorer_class():
    scorer = ReadabilityScorer()
    result = scorer.score(SIMPLE)
    assert result == score_all(SIMPLE)
    assert result["words"] == 15


def test_deterministic():
    assert score_all(COMPLEX) == score_all(COMPLEX)
