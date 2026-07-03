import pytest

from gpt_from_scratch.tokenizer import CharTokenizer


def test_encode_decode_roundtrip():
    tok = CharTokenizer("hello world")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"


def test_vocab_size_matches_unique_chars():
    tok = CharTokenizer("aabbcc")
    assert tok.vocab_size == 3


def test_rejects_empty_corpus():
    with pytest.raises(ValueError):
        CharTokenizer("")


def test_rejects_unknown_char_on_encode():
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError):
        tok.encode("abcz")


def test_rejects_out_of_range_id_on_decode():
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError):
        tok.decode([0, 1, 99])
