import numpy as np
import pytest

from ai_slop_detector.features import FEATURE_NAMES, extract_features


def test_extract_features_returns_all_named_fields():
    fv = extract_features("This is a short test sentence. It has two sentences.")
    arr = fv.to_array()
    assert arr.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(arr))


def test_empty_text_raises():
    with pytest.raises(ValueError):
        extract_features("")
    with pytest.raises(ValueError):
        extract_features("   \n\t  ")


def test_punctuation_only_raises_rather_than_dividing_by_zero():
    # Regex word matcher finds zero words here -- must raise, not return
    # a garbage/NaN feature vector.
    with pytest.raises(ValueError):
        extract_features("!!! ??? ...")


def test_single_sentence_has_zero_stdev():
    fv = extract_features("Just one sentence here with several words in it.")
    assert fv.sentence_len_stdev == 0.0


def test_contractions_detected():
    fv = extract_features("I don't think we're ready. I'm not sure you've seen it.")
    assert fv.contraction_ratio_per_100w > 0


def test_no_contractions_gives_zero_ratio():
    fv = extract_features("The system processes each request in sequence without delay.")
    assert fv.contraction_ratio_per_100w == 0.0


def test_ai_marker_phrases_detected():
    fv = extract_features(
        "Furthermore, it is important to note that this approach is robust. "
        "Moreover, the system can leverage a seamless, holistic strategy."
    )
    assert fv.ai_marker_ratio_per_100w > 0


def test_first_person_ratio_detected():
    fv = extract_features("I went to the store because my car needed gas.")
    assert fv.first_person_ratio_per_100w > 0


def test_filler_slang_detected():
    fv = extract_features("Honestly idk why this happened, gonna figure it out tbh.")
    assert fv.filler_slang_ratio_per_100w > 0


def test_feature_names_length_matches_to_array():
    fv = extract_features("A simple sentence for testing purposes only.")
    assert len(FEATURE_NAMES) == len(fv.to_array())


def test_possessive_s_is_not_counted_as_a_contraction():
    # Regression test: "today's" / "neighbor's" are possessives, not
    # contractions, and must not inflate contraction_ratio_per_100w.
    fv = extract_features("Today's forecast mentions the neighbor's garden looks great.")
    assert fv.contraction_ratio_per_100w == 0.0


def test_unambiguous_s_contractions_are_still_counted():
    fv = extract_features("It's raining and that's unfortunate, but here's the plan.")
    # "it's", "that's", "here's" are all genuine is/has contractions.
    assert fv.contraction_ratio_per_100w > 0


def test_mixed_possessive_and_contraction_counts_only_the_contraction():
    fv_contraction_only = extract_features("It's a nice day outside today.")
    fv_possessive_only = extract_features("Today's weather is nice outside.")
    assert fv_contraction_only.contraction_ratio_per_100w > 0
    assert fv_possessive_only.contraction_ratio_per_100w == 0.0
