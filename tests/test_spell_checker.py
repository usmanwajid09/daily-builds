"""Tests for the edit-distance spell checker."""

import pytest

from spell_checker import SpellChecker
from spell_checker.model import SpellChecker as CheckerFromModel


def test_import_exports_class():
    assert CheckerFromModel is SpellChecker


def test_default_vocab_built():
    checker = SpellChecker()
    assert len(checker.vocab) > 50
    assert checker.total_words > 0
    assert "the" in checker.vocab
    assert "spelling" in checker.vocab
    assert "receive" in checker.vocab


def test_correct_speling():
    assert SpellChecker().correct("speling") == "spelling"


def test_correct_recieve():
    assert SpellChecker().correct("recieve") == "receive"


def test_correct_teh():
    assert SpellChecker().correct("teh") == "the"


def test_correct_known_word_unchanged():
    checker = SpellChecker()
    assert checker.correct("spelling") == "spelling"
    assert checker.correct("the") == "the"
    assert checker.correct("receive") == "receive"


def test_correct_empty_string():
    assert SpellChecker().correct("") == ""


def test_correct_non_string_raises():
    checker = SpellChecker()
    with pytest.raises(TypeError):
        checker.correct(123)
    with pytest.raises(TypeError):
        checker.correct(None)


def test_correct_non_alpha_raises():
    with pytest.raises(ValueError):
        SpellChecker().correct("hello!")
    with pytest.raises(ValueError):
        SpellChecker().correct("well-known")


def test_correct_capitalised_preserves_case():
    checker = SpellChecker()
    assert checker.correct("Teh") == "The"
    assert checker.correct("Speling") == "Spelling"


def test_correct_unknown_word_unchanged():
    # No English word within edit distance 2 of this gibberish.
    assert SpellChecker().correct("zxqv") == "zxqv"


def test_correct_sentence():
    checker = SpellChecker()
    out = checker.correct_sentence("Teh students practise speling.")
    assert out == "The students practise spelling."


def test_correct_sentence_punctuation_preserved():
    checker = SpellChecker()
    out = checker.correct_sentence("Hello, teh world! How are you?")
    assert out == "Hello, the world! How are you?"
    assert checker.correct_sentence("") == ""
    assert checker.correct_sentence("... !!!") == "... !!!"


def test_correct_sentence_non_string_raises():
    with pytest.raises(TypeError):
        SpellChecker().correct_sentence(42)


def test_candidates_prefers_distance_one():
    checker = SpellChecker()
    # "teh" -> "the" is one transpose away, so distance-1 wins.
    assert checker.candidates("teh") == {"the"}


def test_candidates_unknown_returns_itself():
    assert SpellChecker().candidates("zxqv") == {"zxqv"}


def test_candidates_validation():
    checker = SpellChecker()
    with pytest.raises(TypeError):
        checker.candidates(7)
    with pytest.raises(ValueError):
        checker.candidates("")
    with pytest.raises(ValueError):
        checker.candidates("ab12")


def test_frequency_ranking():
    # Custom corpus: "cat" is far more common than "bat", both are
    # one edit from "hat"-like typo "cat" is known; use "dat" instead.
    checker = SpellChecker(corpus={"cat": 50, "bat": 2, "rat": 1})
    assert checker.correct("dat") == "cat"
    assert checker.frequency("cat") == 50
    assert checker.frequency("nope") == 0


def test_custom_corpus_text():
    checker = SpellChecker(corpus="apple banana apple cherry apple")
    assert checker.correct("apple") == "apple"
    assert checker.correct("appl") == "apple"
    assert checker.frequency("apple") == 3


def test_custom_corpus_empty_raises():
    with pytest.raises(ValueError):
        SpellChecker(corpus="")
    with pytest.raises(ValueError):
        SpellChecker(corpus={})


def test_custom_corpus_bad_type_raises():
    with pytest.raises(TypeError):
        SpellChecker(corpus=123)
    with pytest.raises(ValueError):
        SpellChecker(corpus={"ok": 1, "bad word": 2})
    with pytest.raises(ValueError):
        SpellChecker(corpus={"ok": 0})


def test_max_edit_distance_validation():
    with pytest.raises(ValueError):
        SpellChecker(max_edit_distance=3)
    with pytest.raises(ValueError):
        SpellChecker(max_edit_distance=0)


def test_max_edit_distance_one_still_corrects():
    checker = SpellChecker(max_edit_distance=1)
    assert checker.correct("teh") == "the"
    assert checker.correct("speling") == "spelling"


def test_edits1_contains_transpose():
    assert "the" in SpellChecker.edits1("teh")
    assert "receive" in SpellChecker.edits1("recieve")


def test_correct_sentence_numbers_kept():
    checker = SpellChecker()
    assert checker.correct_sentence("teh 42 cats") == "the 42 cats"
