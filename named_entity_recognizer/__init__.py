"""Rule-based named entity recognizer: PERSON / ORG / LOCATION / DATE."""

from named_entity_recognizer.model import (
    DATE,
    LOCATION,
    ORG,
    PERSON,
    NamedEntityRecognizer,
    entities_by_label,
    extract_entities,
)

__all__ = [
    "DATE",
    "LOCATION",
    "ORG",
    "PERSON",
    "NamedEntityRecognizer",
    "entities_by_label",
    "extract_entities",
]
