"""Tests for steganography_lsb.model (LSB image steganography)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from steganography_lsb import (
    capacity_bytes,
    embed_message,
    extract_message,
    lsb_difference,
)


def make_image(shape=(32, 32, 3), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_round_trip_ascii():
    image = make_image()
    stego = embed_message(image, "Hello, world!")
    assert extract_message(stego) == "Hello, world!"


def test_round_trip_unicode():
    image = make_image()
    message = "héllo wörld — 你好 🌍"
    assert extract_message(embed_message(image, message)) == message


def test_round_trip_empty_message():
    image = make_image()
    assert extract_message(embed_message(image, "")) == ""


def test_capacity_calculation():
    # 32x32x3 = 3072 values; header 64 bits -> (3072 - 64) // 8 = 376
    assert capacity_bytes(make_image((32, 32, 3))) == 376
    # 16x16 grayscale = 256 values -> (256 - 64) // 8 = 24
    assert capacity_bytes(make_image((16, 16))) == 24
    # Image too small for the header: capacity is 0, not negative.
    assert capacity_bytes(np.zeros((2, 2), dtype=np.uint8)) == 0


def test_message_too_long_raises():
    image = make_image((8, 8))  # capacity (64 - 64) // 8 = 0
    with pytest.raises(ValueError, match="too long|capacity"):
        embed_message(image, "x")
    image = make_image((16, 16))  # capacity 24 bytes
    with pytest.raises(ValueError):
        embed_message(image, "a" * 25)


def test_extract_from_unmarked_image_raises():
    image = make_image()
    with pytest.raises(ValueError, match="magic"):
        extract_message(image)


def test_tampered_magic_raises():
    image = make_image()
    stego = embed_message(image, "secret")
    tampered = stego.copy()
    # Flip the LSB of the very first value: corrupts the first magic bit.
    tampered.flat[0] ^= 1
    with pytest.raises(ValueError, match="magic"):
        extract_message(tampered)


def test_only_lsbs_change():
    image = make_image()
    stego = embed_message(image, "only LSBs may change")
    assert stego.shape == image.shape
    assert stego.dtype == image.dtype
    diff = np.abs(stego.astype(int) - image.astype(int))
    assert diff.max() <= 1
    # Every changed value differs by exactly 1, i.e. only its LSB flipped.
    assert set(np.unique(diff)) <= {0, 1}
    # The input image itself is untouched.
    assert lsb_difference(image, image) == 0


def test_lsb_difference_counts():
    image = make_image()
    stego = embed_message(image, "count me")
    count = lsb_difference(image, stego)
    assert 0 < count <= image.size
    # Manual recount agrees.
    assert count == int(np.count_nonzero((image & 1) != (stego & 1)))


def test_grayscale_images_work():
    image = make_image((32, 32), seed=7)
    assert extract_message(embed_message(image, "gray")) == "gray"


def test_non_uint8_input_raises():
    float_image = np.zeros((16, 16), dtype=np.float64)
    with pytest.raises(TypeError, match="uint8"):
        embed_message(float_image, "hi")
    with pytest.raises(TypeError, match="uint8"):
        capacity_bytes(np.zeros((16, 16), dtype=np.int32))
    with pytest.raises(TypeError, match="uint8"):
        extract_message(np.zeros((16, 16), dtype=np.uint16))


def test_non_str_message_raises():
    image = make_image()
    with pytest.raises(TypeError, match="str"):
        embed_message(image, b"bytes, not str")
    with pytest.raises(TypeError, match="str"):
        embed_message(image, 123)


def test_embed_at_exact_capacity():
    image = make_image((16, 16))  # capacity 24 bytes
    message = "x" * 24
    assert extract_message(embed_message(image, message)) == message


def test_demo_behaviour():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    stego = embed_message(image, "Hello daily-builds!")
    assert extract_message(stego) == "Hello daily-builds!"
    assert capacity_bytes(image) == (64 * 64 * 3 - 64) // 8
    assert np.abs(stego.astype(int) - image.astype(int)).max() <= 1
