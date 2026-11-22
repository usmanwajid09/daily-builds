"""Tests for template_captioner."""
import numpy as np
import pytest

from template_captioner.captioner import (
    analyze_image,
    caption,
    count_shapes,
    dominant_colours,
    largest_shape_quadrant,
    make_test_image,
)


def test_dominant_colour_of_solid_image():
    img = np.ones((16, 16, 3)) * np.array([1.0, 0.0, 0.0])
    colours = dominant_colours(img)
    assert colours[0][0] == "red"
    assert abs(colours[0][1] - 1.0) < 1e-9


def test_dominant_colours_black_and_white():
    img = np.zeros((16, 16, 3))
    img[:8] = 1.0  # top half white
    colours = dict(dominant_colours(img))
    assert abs(colours["white"] - 0.5) < 1e-9
    assert abs(colours["black"] - 0.5) < 1e-9


def test_dominant_colours_min_fraction_filters_rare_colours():
    img = np.zeros((16, 16, 3))
    img[0, 0] = [1.0, 0.0, 0.0]  # a single red pixel
    names = [n for n, _ in dominant_colours(img, min_fraction=0.05)]
    assert "red" not in names
    assert "black" in names


def test_dominant_colours_rejects_bad_fraction():
    with pytest.raises(ValueError):
        dominant_colours(np.zeros((8, 8, 3)), min_fraction=0)


def test_count_shapes_single_circle():
    img = make_test_image(32, shapes=[("circle", (1.0, 0.0, 0.0), (16, 16), 6)])
    n, sizes = count_shapes(img)
    assert n == 1
    assert sizes[0] > 100  # pi * 6^2 ~= 113


def test_count_shapes_two_separate_shapes():
    img = make_test_image(32, shapes=[
        ("circle", (1.0, 0.0, 0.0), (8, 8), 4),
        ("square", (0.0, 0.0, 1.0), (24, 24), 4)], background=1.0)
    n, sizes = count_shapes(img)
    assert n == 2
    assert sizes == sorted(sizes, reverse=True)


def test_count_shapes_blank_is_zero():
    n, sizes = count_shapes(np.zeros((32, 32, 3)))
    assert n == 0
    assert sizes == []


def test_count_shapes_rejects_bad_min_size():
    with pytest.raises(ValueError):
        count_shapes(np.zeros((8, 8, 3)), min_size=0)


def test_largest_shape_quadrant_upper_left():
    img = make_test_image(32, shapes=[("square", (1.0, 1.0, 1.0), (8, 8), 4)])
    assert largest_shape_quadrant(img) == "upper left"


def test_largest_shape_quadrant_lower_right():
    img = make_test_image(32, shapes=[("square", (1.0, 1.0, 1.0), (25, 25), 4)])
    assert largest_shape_quadrant(img) == "lower right"


def test_largest_shape_quadrant_none_when_blank():
    assert largest_shape_quadrant(np.zeros((16, 16, 3))) is None


def test_caption_blank_image():
    text = caption(np.zeros((16, 16, 3)))
    assert isinstance(text, str) and text
    assert "no distinct shapes" in text
    assert "black" in text


def test_caption_red_circle_mentions_colour_and_count():
    img = make_test_image(32, shapes=[("circle", (1.0, 0.0, 0.0), (16, 16), 6)])
    text = caption(img)
    assert "1 shape" in text
    assert "red" in text
    assert "dark" in text


def test_caption_two_shapes_mentions_both_colours():
    img = make_test_image(32, shapes=[
        ("square", (0.0, 0.0, 1.0), (8, 8), 4),
        ("circle", (1.0, 1.0, 0.0), (24, 24), 5)], background=1.0)
    text = caption(img)
    assert "2 shapes" in text
    assert "blue" in text and "yellow" in text
    assert "light" in text


def test_analyze_image_keys_and_ranges():
    img = make_test_image(32, shapes=[("circle", (0.0, 0.8, 0.0), (16, 16), 10)])
    info = analyze_image(img)
    assert set(info) == {"brightness", "contrast", "saturation", "background",
                         "dominant_colours", "n_shapes", "shape_sizes",
                         "largest_shape_quadrant", "is_colourful",
                         "brightness_word"}
    assert 0.0 <= info["brightness"] <= 1.0
    assert info["contrast"] >= 0.0
    assert info["n_shapes"] == 1
    assert info["is_colourful"] is True


def test_grayscale_input_accepted():
    grey = np.zeros((16, 16))
    grey[4:12, 4:12] = 1.0
    info = analyze_image(grey)
    assert info["n_shapes"] == 1


def test_uint8_input_accepted():
    img = (np.zeros((16, 16, 3), dtype=np.uint8))
    img[4:12, 4:12] = [255, 0, 0]
    assert caption(img)  # non-empty caption
    assert "red" in [n for n, _ in dominant_colours(img)]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        analyze_image(np.zeros((8, 8, 4)))  # 4 channels
    with pytest.raises(ValueError):
        analyze_image(np.zeros((0, 8, 3)))  # empty
    with pytest.raises(ValueError):
        analyze_image(np.full((8, 8, 3), 2.0))  # float out of [0, 1]
    with pytest.raises(ValueError):
        analyze_image(np.full((8, 8, 3), np.nan))


def test_make_test_image_validation():
    with pytest.raises(ValueError):
        make_test_image(32, shapes=[("triangle", (1, 0, 0), (16, 16), 4)])
    with pytest.raises(ValueError):
        make_test_image(0)
    with pytest.raises(ValueError):
        make_test_image(32, background=2.0)
