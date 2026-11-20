"""Template-matching OCR for printed digits and letters (numpy only)."""

from ocr_template.model import (
    CHARACTERS,
    TEMPLATES,
    TemplateOCR,
    classify_character,
    recognize,
    render_text,
    segment_characters,
)

__all__ = [
    "CHARACTERS",
    "TEMPLATES",
    "TemplateOCR",
    "classify_character",
    "recognize",
    "render_text",
    "segment_characters",
]
