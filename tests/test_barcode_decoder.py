"""Tests for barcode_decoder.barcode (encode/render/decode round-trip)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from barcode_decoder.barcode import (
    ALPHABET,
    barcode_width,
    decode,
    encode,
    render_barcode,
)


def test_round_trip_hello():
    assert decode(render_barcode("HELLO")) == "HELLO"


def test_round_trip_abc123():
    assert decode(render_barcode("ABC123")) == "ABC123"


def test_round_trip_single_char():
    assert decode(render_barcode("9")) == "9"


def test_round_trip_with_space():
    assert decode(render_barcode("A B")) == "A B"


def test_render_shape_and_width_match_barcode_width():
    image = render_barcode("HELLO 42")
    assert image.shape == (40, barcode_width("HELLO 42"))
    assert image.shape[1] == barcode_width("HELLO 42", unit_width=3)
    assert render_barcode("HELLO 42", unit_width=5, height=20).shape == (
        20, barcode_width("HELLO 42", unit_width=5))


def test_render_contains_only_black_and_white():
    image = render_barcode("ABC123")
    assert set(np.unique(image).tolist()) <= {0, 255}
    assert 0 in np.unique(image) and 255 in np.unique(image)


def test_decode_all_white_image_raises():
    with pytest.raises(ValueError, match="no barcode"):
        decode(np.full((40, 100), 255, dtype=np.uint8))


def test_decode_corrupted_start_sentinel_raises():
    image = render_barcode("HELLO")
    # Start sentinel starts after the 10-unit quiet zone (30 px at
    # unit_width=3); its bars are 6 px wide. Paint its 2nd (white) bar
    # black, merging bars 1-3 into one 6-unit black run.
    image[:, 36:42] = 0
    with pytest.raises(ValueError):
        decode(image)


def test_encode_rejects_unsupported_chars_and_empty_string():
    with pytest.raises(ValueError, match="unsupported character 'h'"):
        encode("hello")
    with pytest.raises(ValueError, match="unsupported character '!'"):
        encode("A!B")
    with pytest.raises(ValueError, match="must not be empty"):
        encode("")


def test_invalid_unit_width_and_height_raise():
    with pytest.raises(ValueError, match="unit_width"):
        render_barcode("A", unit_width=0)
    with pytest.raises(ValueError, match="unit_width"):
        render_barcode("A", unit_width=2.5)
    with pytest.raises(ValueError, match="height"):
        render_barcode("A", height=0)
    with pytest.raises(ValueError, match="unit_width"):
        barcode_width("A", unit_width=-1)


def test_decode_rejects_3d_and_non_array_input():
    with pytest.raises(ValueError, match="2D"):
        decode(np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="2D"):
        decode(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(ValueError, match="2D"):
        decode("not an image")


def test_round_trip_every_alphabet_character():
    for char in ALPHABET:
        assert decode(render_barcode(char)) == char


def test_round_trip_with_unit_width_1():
    assert decode(render_barcode("HELLO 42", unit_width=1)) == "HELLO 42"
    assert decode(render_barcode("Z", unit_width=1, height=10)) == "Z"
