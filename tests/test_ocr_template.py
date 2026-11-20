import numpy as np
import pytest

from ocr_template import (
    CHARACTERS,
    TEMPLATES,
    TemplateOCR,
    classify_character,
    recognize,
    render_text,
    segment_characters,
)


def test_all_templates_have_correct_shape():
    assert len(TEMPLATES) == 37  # 10 digits + 26 letters + space
    for ch, glyph in TEMPLATES.items():
        assert glyph.shape == (7, 5), ch
        assert set(np.unique(glyph)) <= {0.0, 1.0}


def test_render_and_recognize_digits():
    ocr = TemplateOCR()
    for text in ("0123456789", "2026", "007"):
        assert ocr.recognize(ocr.render(text)) == text


def test_render_and_recognize_letters():
    ocr = TemplateOCR()
    assert ocr.recognize(ocr.render("HELLO")) == "HELLO"
    assert ocr.recognize(ocr.render("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) == (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_recognize_mixed_text_with_spaces():
    ocr = TemplateOCR()
    assert ocr.recognize(ocr.render("HELLO 123 WORLD")) == "HELLO 123 WORLD"


def test_recognize_scaled_text():
    ocr = TemplateOCR()
    for scale in (1, 2, 3):
        assert ocr.recognize(ocr.render("GTA 6", scale=scale)) == "GTA 6"


def test_classify_every_character_exactly():
    for ch in CHARACTERS:
        found, dist = classify_character(TEMPLATES[ch])
        assert found == ch
        assert dist == 0


def test_segment_character_count():
    image = render_text("ABC")
    cells = segment_characters(image)
    assert len(cells) == 3
    for cell in cells:
        assert cell.shape == (7, 5)


def test_segment_word_space_produces_blank_cell():
    image = render_text("A B")
    cells = segment_characters(image)
    assert len(cells) == 3
    assert cells[1].sum() == 0


def test_recognize_tolerates_light_noise():
    ocr = TemplateOCR()
    rng = np.random.default_rng(4)
    image = ocr.render("HELLO 123", scale=2)
    noisy = image.copy()
    flips = rng.choice(image.size, size=12, replace=False)
    noisy.ravel()[flips] = 1.0 - noisy.ravel()[flips]
    assert ocr.recognize(noisy) == "HELLO 123"


def test_recognize_with_scores_reports_zero_distance_on_clean_text():
    ocr = TemplateOCR()
    _, scored = ocr.recognize_with_scores(ocr.render("HI"))
    assert [d for _, d in scored] == [0, 0]


def test_render_unsupported_character_raises():
    with pytest.raises(ValueError):
        render_text("hello")  # lowercase is not in the font


def test_render_empty_text_raises():
    with pytest.raises(ValueError):
        render_text("")


def test_render_invalid_scale_raises():
    with pytest.raises(ValueError):
        render_text("A", scale=0)


def test_segment_blank_image_raises():
    with pytest.raises(ValueError):
        segment_characters(np.zeros((7, 20)))


def test_recognize_non_2d_raises():
    with pytest.raises(ValueError):
        recognize(np.zeros((7, 5, 3)))


def test_recognize_leading_trailing_margin():
    ocr = TemplateOCR()
    image = ocr.render("AB")
    padded = np.pad(image, ((2, 2), (4, 6)))
    assert ocr.recognize(padded) == "AB"


def test_templates_are_distinct():
    glyphs = [TEMPLATES[ch] for ch in CHARACTERS]
    for i, a in enumerate(glyphs):
        for b in glyphs[i + 1:]:
            assert int(np.sum(a != b)) >= 1
