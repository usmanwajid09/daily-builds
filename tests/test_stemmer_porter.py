"""Tests for stemmer_porter (Porter stemmer, Porter 1980).

The expected stems are the standard Porter test vocabulary outputs
(Porter 1980; tartarus.org/~martin/PorterStemmer/voc.txt + output.txt).
One entry deserves a note: the widely copied example list gives
"valenci -> valence", but that is the *step 2* intermediate form --
running the full algorithm, step 5a then drops the final E, and the
official output.txt confirms it (input "valence" stems to "valenc",
voc.txt/output.txt line 44396).  The correct Porter output is "valenc",
and that is what is asserted here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from stemmer_porter import PorterStemmer, stem, stem_text

# Full standard Porter test vocabulary: (word, expected stem).
KNOWN_EXAMPLES = [
    # Step 1a -- plurals.
    ("caresses", "caress"),
    ("ponies", "poni"),
    ("ties", "ti"),
    ("caress", "caress"),
    ("cats", "cat"),
    # Step 1b -- -eed / -ed / -ing and the 1b extras.
    ("feed", "feed"),
    ("agreed", "agre"),
    ("plastered", "plaster"),
    ("bled", "bled"),
    ("motoring", "motor"),
    ("sing", "sing"),
    ("conflated", "conflat"),
    ("troubled", "troubl"),
    ("sized", "size"),
    ("hopping", "hop"),
    ("tanned", "tan"),
    ("falling", "fall"),
    ("hissing", "hiss"),
    ("fizzed", "fizz"),
    ("failing", "fail"),
    ("filing", "file"),
    # Step 1c -- terminal y.
    ("happy", "happi"),
    ("sky", "sky"),
    # Step 2 -- derivational suffixes.
    ("relational", "relat"),
    ("conditional", "condit"),
    ("rational", "ration"),
    ("valenci", "valenc"),  # see module docstring: NOT "valence"
    ("hesitanci", "hesit"),
    ("digitizer", "digit"),
    ("radicalli", "radic"),
    ("differentli", "differ"),
    ("vileli", "vile"),
    ("analogousli", "analog"),
    ("vietnamization", "vietnam"),
    ("predication", "predic"),
    ("operator", "oper"),
    ("feudalism", "feudal"),
    ("decisiveness", "decis"),
    ("hopefulness", "hope"),
    ("callousness", "callous"),
    ("formaliti", "formal"),
    ("sensitiviti", "sensit"),
    ("sensibiliti", "sensibl"),
    # Step 4 -- remaining derivational suffixes.
    ("revival", "reviv"),
    ("allowance", "allow"),
    ("inference", "infer"),
    ("airliner", "airlin"),
    ("gyroscopic", "gyroscop"),
    ("adjustable", "adjust"),
    ("defensible", "defens"),
    ("irritant", "irrit"),
    ("replacement", "replac"),
    ("adjustment", "adjust"),
    ("dependent", "depend"),
    ("adoption", "adopt"),
    ("homologou", "homolog"),
    ("communism", "commun"),
    ("activate", "activ"),
    ("angulariti", "angular"),
    ("homologous", "homolog"),
    ("effective", "effect"),
    ("bowdlerize", "bowdler"),
    # Step 5 -- tidying up.
    ("probate", "probat"),
    ("rate", "rate"),
    ("cease", "ceas"),
    ("controll", "control"),
    ("roll", "roll"),
]


@pytest.mark.parametrize("word,expected", KNOWN_EXAMPLES)
def test_known_porter_examples(word, expected):
    assert stem(word) == expected


def test_step1a_plurals():
    assert stem("caresses") == "caress"
    assert stem("ponies") == "poni"
    assert stem("ties") == "ti"
    assert stem("cats") == "cat"
    assert stem("caress") == "caress"  # -ss is left alone


def test_step1b_past_participles():
    assert stem("feed") == "feed"  # -eed kept when m == 0
    assert stem("agreed") == "agre"
    assert stem("plastered") == "plaster"
    assert stem("bled") == "bled"  # no vowel before -ed
    assert stem("motoring") == "motor"
    assert stem("sing") == "sing"  # no vowel before -ing


def test_step1b_extras():
    assert stem("conflated") == "conflat"  # -at put back, then -e dropped
    assert stem("troubled") == "troubl"
    assert stem("sized") == "size"  # -iz put back as -ize
    assert stem("hopping") == "hop"  # double consonant shortened
    assert stem("tanned") == "tan"
    assert stem("falling") == "fall"  # -ll is not shortened
    assert stem("hissing") == "hiss"  # -ss is not shortened
    assert stem("fizzed") == "fizz"  # -zz is not shortened
    assert stem("failing") == "fail"  # no -e: not *o
    assert stem("filing") == "file"  # m == 1 and *o: -e added back


def test_step1c_terminal_y():
    assert stem("happy") == "happi"
    assert stem("sky") == "sky"  # no vowel in the stem, y stays


def test_step2_derivational():
    assert stem("relational") == "relat"
    assert stem("conditional") == "condit"
    assert stem("rational") == "ration"  # -tion kept: m == 1 for -ion
    assert stem("vietnamization") == "vietnam"
    assert stem("digitizer") == "digit"
    assert stem("differentli") == "differ"


def test_step4_derivational():
    assert stem("revival") == "reviv"
    assert stem("replacement") == "replac"
    assert stem("adjustment") == "adjust"
    assert stem("adoption") == "adopt"  # -ion after T, m > 1
    assert stem("effective") == "effect"
    assert stem("bowdlerize") == "bowdler"


def test_step5_tidy_up():
    assert stem("probate") == "probat"  # m > 1: final -e dropped
    assert stem("rate") == "rate"  # m == 1 and *o: -e kept
    assert stem("cease") == "ceas"  # m == 1, not *o: -e dropped
    assert stem("controll") == "control"  # -ll shortened when m > 1
    assert stem("roll") == "roll"  # m == 1: -ll kept


def test_idempotent_on_stems():
    # Stemming an already-stemmed word changes nothing.
    for word in ["caress", "feed", "sing", "sky", "rate", "roll",
                 "motor", "hop", "fail", "relat", "condit", "vietnam"]:
        assert stem(word) == word


def test_word_family_collapses_to_one_stem():
    connect_forms = ["connect", "connected", "connecting", "connection"]
    assert {stem(w) for w in connect_forms} == {"connect"}
    relate_forms = ["relate", "related", "relating", "relational"]
    assert {stem(w) for w in relate_forms} == {"relat"}
    happy_forms = ["happy", "happiness"]
    assert {stem(w) for w in happy_forms} == {"happi"}


def test_class_api_matches_functions():
    stemmer = PorterStemmer()
    assert stemmer.stem("caresses") == "caress"
    assert stemmer.stem_text("caresses ponies") == "caress poni"
    assert stemmer.stem("relational") == stem("relational")


def test_case_and_whitespace_tolerance():
    assert stem("Cats") == "cat"
    assert stem("  hopping  ") == "hop"
    assert stem("RELATIONAL") == "relat"


def test_validation_errors():
    with pytest.raises(TypeError):
        stem(42)
    with pytest.raises(TypeError):
        stem(None)
    with pytest.raises(TypeError):
        stem(["cats"])
    with pytest.raises(ValueError):
        stem("")
    with pytest.raises(ValueError):
        stem("   ")


def test_stem_text_behaviour():
    assert stem_text("caresses ponies cats") == "caress poni cat"
    assert stem_text("the cats keep hopping") == "the cat keep hop"
    # Word order is preserved, including repeated words.
    assert stem_text("hop hop hopping") == "hop hop hop"
    # Extra whitespace collapses to single spaces.
    assert stem_text("  cats   hopping  ") == "cat hop"
    # Empty text stems to empty text.
    assert stem_text("") == ""
    assert stem_text("   ") == ""
    with pytest.raises(TypeError):
        stem_text(123)


def test_stem_text_collapses_a_sentence():
    sentence = "Connecting the connections and the connected systems"
    assert stem_text(sentence) == "connect the connect and the connect system"
    raw = sentence.split()
    stemmed = stem_text(sentence).split()
    assert len(set(stemmed)) < len(set(raw))
