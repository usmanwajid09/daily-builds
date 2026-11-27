"""Tests for the plagiarism / text-overlap detector."""

import pytest

from plagiarism_detector import (
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

A = ("The quick brown fox jumps over the lazy dog near the river bank "
     "on a bright sunny morning in the month of June.")
B = ("The quick brown fox jumps over the lazy dog near the river bank "
     "on a bright sunny morning in the month of July.")
C = ("Quantum computers use qubits and superposition to perform certain "
     "calculations far faster than classical machines ever could.")


def test_identical_texts_score_one():
    assert jaccard(A, A) == 1.0
    assert containment(A, A) == 1.0
    assert lcs_ratio(A, A) == 1.0


def test_near_copy_high_similarity():
    assert jaccard(A, B) > 0.7
    assert containment(A, B) > 0.8
    assert lcs_ratio(A, B) > 0.9


def test_unrelated_texts_low_similarity():
    assert jaccard(A, C) == 0.0
    assert containment(A, C) == 0.0
    assert lcs_ratio(A, C) < 0.1


def test_compare_report_fields():
    report = compare(A, B)
    for key in ("jaccard", "containment", "lcs_ratio", "combined_score",
                "verdict", "shared_shingles", "top_shingles", "n_gram"):
        assert key in report
    assert report["verdict"] == "high overlap — likely copied"
    assert report["shared_shingles"] > 10


def test_compare_unrelated_verdict():
    assert compare(A, C)["verdict"] == "no meaningful overlap"


def test_containment_catches_short_in_long():
    short = "the quick brown fox jumps over the lazy dog"
    long_text = ("In a story I once read, the quick brown fox jumps over "
                 "the lazy dog while the sun sets over the hills.")
    assert containment(short, long_text) == 1.0
    assert jaccard(short, long_text) < 1.0


def test_shingles_n_gram_size():
    assert shingles("one two three four", n=2) == {
        "one two", "two three", "three four"}
    assert shingles("one two", n=3) == set()


def test_shingles_case_insensitive():
    assert shingles("The Cat Sat") == shingles("the cat sat")


def test_overlap_shingles_lists_shared():
    shared = overlap_shingles(A, B)
    assert "the quick brown" in shared
    assert "quantum computers use" not in shared


def test_lcs_reordered_text():
    a = "one two three four five six"
    b = "six five four three two one"
    assert 0 < lcs_ratio(a, b) < 0.5


def test_verdict_bands():
    assert verdict(0.9) == "high overlap — likely copied"
    assert verdict(0.3) == "moderate overlap — possible paraphrase"
    assert verdict(0.1) == "low overlap — mostly original"
    assert verdict(0.0) == "no meaningful overlap"


def test_verdict_validation():
    with pytest.raises(ValueError):
        verdict(1.5)
    with pytest.raises(ValueError):
        verdict("high")


def test_tokenize():
    assert tokenize("Hello, World 123!") == ["hello", "world", "123"]


def test_empty_text_raises():
    with pytest.raises(ValueError):
        compare("", A)
    with pytest.raises(ValueError):
        compare(A, "   ")


def test_non_string_raises():
    with pytest.raises(ValueError):
        jaccard(42, A)


def test_bad_n():
    with pytest.raises(ValueError):
        shingles(A, n=0)
    with pytest.raises(ValueError):
        compare(A, B, n=-1)


def test_compare_many():
    reports = compare_many([A, B, C])
    assert len(reports) == 3  # 3 choose 2
    assert {(r["i"], r["j"]) for r in reports} == {(0, 1), (0, 2), (1, 2)}
    assert reports[0]["combined_score"] > reports[1]["combined_score"]


def test_compare_many_validation():
    with pytest.raises(ValueError):
        compare_many([A])
    with pytest.raises(ValueError):
        compare_many("not a list")


def test_detector_class():
    detector = PlagiarismDetector(n=2)
    report = detector.compare(A, B)
    assert report["n_gram"] == 2
    assert report["combined_score"] > 0.5
    assert len(detector.compare_many([A, B, C])) == 3


def test_detector_bad_n():
    with pytest.raises(ValueError):
        PlagiarismDetector(n=0)
