import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from language_detector.demo import DEMO_SENTENCES
from language_detector.model import (
    LANGUAGES,
    SAMPLE_TEXTS,
    LanguageDetector,
    detect_language,
    normalize,
    profile_distance,
    trigram_profile,
    trigrams,
)

DEMO_EXPECTED = {
    "en": "The sun rises early in the summer and the birds sing in the trees behind our house every morning.",
    "fr": "Le soleil se lève tôt en été et les oiseaux chantent dans les arbres derrière notre maison chaque matin.",
    "es": "El sol sale temprano en verano y los pájaros cantan en los árboles detrás de nuestra casa cada mañana.",
    "de": "Die Sonne geht im Sommer früh auf und die Vögel singen jeden Morgen in den Bäumen hinter unserem Haus.",
}


@pytest.fixture
def detector():
    return LanguageDetector()


# --- demo behaviour: one sentence per language ---

def test_demo_sentences_classify_correctly(detector):
    for code, sentence in DEMO_SENTENCES.items():
        assert detector.classify(sentence) == code


def test_demo_module_sentences_match_expected():
    assert DEMO_SENTENCES == DEMO_EXPECTED


def test_training_sentences_classify_correctly(detector):
    # Every embedded training sentence should be recognised as its own language.
    for code, texts in SAMPLE_TEXTS.items():
        for text in texts:
            assert detector.classify(text) == code


def test_demo_script_runs_and_reports_4_out_of_4():
    result = subprocess.run(
        [sys.executable, "language_detector/demo.py"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Demo accuracy: 4/4" in result.stdout


# --- validation errors ---

def test_empty_string_raises_value_error(detector):
    with pytest.raises(ValueError, match="must not be empty"):
        detector.classify("")


def test_whitespace_only_raises_value_error(detector):
    with pytest.raises(ValueError, match="must not be empty"):
        detector.classify("   \n\t  ")


def test_non_string_raises_type_error(detector):
    with pytest.raises(TypeError, match="must be a string"):
        detector.classify(123)
    with pytest.raises(TypeError, match="must be a string"):
        detector.classify(None)


def test_distances_non_string_raises_type_error(detector):
    with pytest.raises(TypeError):
        detector.distances(["not", "a", "string"])


def test_distances_empty_raises_value_error(detector):
    with pytest.raises(ValueError):
        detector.distances("")


def test_invalid_top_n_raises(detector):
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        LanguageDetector(top_n=0)
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        LanguageDetector(top_n=-5)


def test_invalid_min_length_raises():
    with pytest.raises(ValueError, match="min_length must be a positive integer"):
        LanguageDetector(min_length=0)


def test_fit_rejects_unknown_language_code(detector):
    with pytest.raises(ValueError, match="unsupported language code"):
        detector.fit({"xx": ["some text here for testing"]})


def test_fit_rejects_empty_training_data(detector):
    with pytest.raises(ValueError, match="must not be empty"):
        detector.fit({})


def test_fit_rejects_non_dict(detector):
    with pytest.raises(TypeError, match="must be a dict"):
        detector.fit([("en", ["hello world"])])


def test_fit_rejects_empty_text_list(detector):
    with pytest.raises(ValueError, match="non-empty list"):
        detector.fit({"en": []})


# --- edge cases: unknown / short text ---

def test_short_text_returns_unknown(detector):
    assert detector.classify("hi") == "unknown"
    assert detector.classify("hello") == "unknown"
    assert detector.classify("bonjour") == "unknown"


def test_digits_only_returns_unknown(detector):
    assert detector.classify("1234567890") == "unknown"


def test_punctuation_only_returns_unknown(detector):
    # No letters at all means too little evidence, so classify() says
    # "unknown"; distances() on the same text raises instead.
    assert detector.classify("!!! ??? ...") == "unknown"
    with pytest.raises(ValueError):
        detector.distances("!!! ??? ...")


def test_case_and_punctuation_do_not_change_result(detector):
    sentence = DEMO_SENTENCES["de"]
    assert detector.classify(sentence) == "de"
    assert detector.classify(sentence.upper()) == "de"
    assert detector.classify(sentence + "!!!") == "de"


# --- scores / distances ---

def test_distances_cover_all_languages_and_best_matches_classify(detector):
    for code, sentence in DEMO_SENTENCES.items():
        dists = detector.distances(sentence)
        assert set(dists) == set(LANGUAGES)
        assert all(isinstance(v, (int, float)) and v >= 0 for v in dists.values())
        assert min(dists, key=dists.get) == code == detector.classify(sentence)


def test_scores_sum_to_one_and_argmax_matches_classify(detector):
    for code, sentence in DEMO_SENTENCES.items():
        scores = detector.scores(sentence)
        assert set(scores) == set(LANGUAGES)
        assert np.isclose(sum(scores.values()), 1.0)
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        assert max(scores, key=scores.get) == code


def test_scores_raise_on_empty_text(detector):
    with pytest.raises(ValueError):
        detector.scores("")


# --- helpers and retraining ---

def test_normalize_lowercases_and_strips_digits():
    assert normalize("Hello, WORLD! 123") == "hello world"


def test_normalize_non_string_raises_type_error():
    with pytest.raises(TypeError):
        normalize(42)


def test_trigrams_include_word_boundaries():
    assert trigrams("abc") == [" ab", "abc", "bc "]


def test_trigram_profile_is_ranked_and_bounded():
    # "at " occurs 4 times (cat, sat, mat, cat) — more than any other
    # trigram — so it must rank first; top_n caps the profile length.
    profile = trigram_profile("the cat sat on the mat the cat", top_n=5)
    assert len(profile) == 5
    assert profile[0] == "at "


def test_profile_distance_is_zero_for_identical_profiles():
    profile = [" th", "the", "he "]
    assert profile_distance(profile, profile, top_n=3) == 0


def test_profile_distance_penalises_missing_trigrams():
    dist = profile_distance(["zzz", "qqq"], [" th", "the"], top_n=10)
    assert dist == 20


def test_language_name_lookup(detector):
    assert detector.language_name("en") == "English"
    assert detector.language_name("fr") == "French"
    assert detector.language_name("es") == "Spanish"
    assert detector.language_name("de") == "German"
    assert detector.language_name("unknown") == "unknown"
    with pytest.raises(ValueError, match="unsupported language code"):
        detector.language_name("xx")


def test_detect_language_convenience_function():
    assert detect_language(DEMO_SENTENCES["es"]) == "es"
    assert detect_language("hi") == "unknown"


def test_fit_retrains_on_subset(detector):
    detector.fit({
        "en": SAMPLE_TEXTS["en"],
        "fr": SAMPLE_TEXTS["fr"],
    })
    assert detector.languages == ["en", "fr"]
    assert detector.classify(DEMO_SENTENCES["en"]) == "en"
    assert detector.classify(DEMO_SENTENCES["fr"]) == "fr"
    assert "es" not in detector.distances(DEMO_SENTENCES["en"])
