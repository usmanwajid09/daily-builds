"""Tests for the rule-based named entity recognizer."""

import pytest

from named_entity_recognizer import (
    DATE,
    LOCATION,
    ORG,
    PERSON,
    NamedEntityRecognizer,
    entities_by_label,
    extract_entities,
)

TEXT = (
    "Usman Wajid, a student at FAST-NUCES in Lahore, presented his "
    "project to Google on 12 March 2026. "
    "Ahmed Khan from Microsoft Corporation met Sara in Karachi, "
    "Pakistan on 2026-04-01."
)


def _labels(text):
    return {(e["text"], e["label"]) for e in extract_entities(text)}


def test_person_two_word_name():
    entities = _labels("Usman Wajid arrived.")
    assert ("Usman Wajid", PERSON) in entities


def test_person_first_name_gazetteer():
    entities = _labels("Ahmed arrived early.")
    assert ("Ahmed", PERSON) in entities


def test_org_suffix():
    entities = _labels("She works at Microsoft Corporation.")
    assert ("Microsoft Corporation", ORG) in entities


def test_org_university():
    entities = _labels("He studies at FAST-NUCES University.")
    assert ("FAST-NUCES University", ORG) in entities


def test_location_city():
    entities = _labels("They travelled to Lahore yesterday.")
    assert ("Lahore", LOCATION) in entities


def test_location_country():
    entities = _labels("She lives in Pakistan.")
    assert ("Pakistan", LOCATION) in entities


def test_date_day_month_year():
    entities = _labels("The meeting is on 12 March 2026.")
    assert ("12 March 2026", DATE) in entities


def test_date_month_day_year():
    entities = _labels("The meeting is on March 12, 2026.")
    assert ("March 12, 2026", DATE) in entities


def test_date_iso():
    entities = _labels("The deadline is 2026-04-01.")
    assert ("2026-04-01", DATE) in entities


def test_date_slash():
    entities = _labels("The deadline is 12/03/2026.")
    assert ("12/03/2026", DATE) in entities


def test_invalid_date_not_extracted():
    entities = extract_entities("The date is 45 Foo 2026.")
    assert all(e["label"] != DATE for e in entities)


def test_offsets_correct():
    text = "Usman Wajid went to Lahore."
    entities = extract_entities(text)
    for entity in entities:
        assert text[entity["start"]:entity["end"]] == entity["text"]


def test_entities_sorted_by_position():
    entities = extract_entities(TEXT)
    starts = [e["start"] for e in entities]
    assert starts == sorted(starts)


def test_full_text_coverage():
    entities = _labels(TEXT)
    assert ("Usman Wajid", PERSON) in entities
    assert ("Lahore", LOCATION) in entities
    assert ("12 March 2026", DATE) in entities
    assert ("Ahmed Khan", PERSON) in entities
    assert ("Microsoft Corporation", ORG) in entities
    assert ("Karachi", LOCATION) in entities
    assert ("Pakistan", LOCATION) in entities
    assert ("2026-04-01", DATE) in entities


def test_sentence_starter_not_entity():
    entities = extract_entities("The dog ran fast.")
    assert entities == []


def test_empty_text_raises():
    with pytest.raises(ValueError):
        extract_entities("")
    with pytest.raises(ValueError):
        extract_entities("   ")


def test_non_string_raises():
    with pytest.raises(ValueError):
        extract_entities(42)


def test_no_entities_returns_empty():
    assert extract_entities("the cat sat on the mat.") == []


def test_entities_by_label_groups():
    grouped = entities_by_label(TEXT)
    assert "Usman Wajid" in grouped[PERSON]
    assert "Lahore" in grouped[LOCATION]
    assert "12 March 2026" in grouped[DATE]
    assert "Microsoft Corporation" in grouped[ORG]
    assert set(grouped) == {PERSON, ORG, LOCATION, DATE}


def test_recognizer_class():
    recognizer = NamedEntityRecognizer()
    entities = recognizer.extract("Sara went to London on 1 January 2027.")
    texts = {e["text"] for e in entities}
    assert "Sara" in texts
    assert "London" in texts
    assert "1 January 2027" in texts
    assert recognizer.by_label("Sara went to London.")["PERSON"] == ["Sara"]


def test_deterministic():
    assert extract_entities(TEXT) == extract_entities(TEXT)
