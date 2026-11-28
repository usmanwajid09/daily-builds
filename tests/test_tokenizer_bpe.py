"""Tests for the BPE tokenizer (tokenizer_bpe)."""

import pytest

from tokenizer_bpe import BPETokenizer
from tokenizer_bpe.demo import CORPUS

SMALL = [
    "the cat sat on the mat",
    "the cat likes the dog",
    "the dog chased the cat",
    "machine learning is fun",
    "learning machine learning",
]


def _fit(corpus=None, **kwargs):
    return BPETokenizer(**kwargs).fit(corpus if corpus is not None else SMALL)


def test_round_trip_on_training_text():
    tok = _fit()
    for text in SMALL:
        assert tok.decode(tok.encode(text)) == text


def test_round_trip_on_demo_corpus():
    tok = BPETokenizer(num_merges=50, min_frequency=2).fit(CORPUS)
    for text in CORPUS:
        assert tok.decode(tok.encode(text)) == text


def test_frequent_word_fewer_tokens_than_characters():
    tok = BPETokenizer(num_merges=50, min_frequency=2).fit(CORPUS)
    tokens = tok.encode("learning")
    assert len(tokens) < len("learning") + 1  # +1 for the end-of-word marker
    assert len(tokens) < len("learning")


def test_first_merge_is_most_frequent_pair():
    # 'abab' x3: pair ('a', 'b') occurs 6 times, ('b', 'a') and
    # ('b', '</w>') only 3 times each, so ('a', 'b') must merge first.
    tok = BPETokenizer(num_merges=5, min_frequency=1).fit(
        ["abab abab", "abab"]
    )
    assert tok.merges[0] == ("a", "b")


def test_vocab_grows_with_merges():
    few = _fit(num_merges=1, min_frequency=1)
    many = _fit(num_merges=30, min_frequency=1)
    assert many.vocab_size > few.vocab_size
    assert len(many.merges) > len(few.merges)


def test_unseen_word_encodes_to_characters():
    tok = _fit()
    tokens = tok.encode("zebra")
    assert tokens  # no crash, and something comes back
    assert tok.decode(tokens) == "zebra"
    # An unseen word is mostly characters, not one merged token.
    assert len(tokens) >= len("zebra")


def test_deterministic_fits():
    a = _fit(num_merges=20, min_frequency=2)
    b = _fit(num_merges=20, min_frequency=2)
    assert a.merges == b.merges
    assert a.vocab == b.vocab
    assert a.encode("the cat is learning") == b.encode("the cat is learning")


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        BPETokenizer().fit([])
    with pytest.raises(ValueError):
        BPETokenizer().fit(["   ", "!!!"])
    with pytest.raises(ValueError):
        BPETokenizer().fit("")


def test_invalid_hyperparameters_raise():
    with pytest.raises(ValueError):
        BPETokenizer(num_merges=0)
    with pytest.raises(ValueError):
        BPETokenizer(min_frequency=0)
    with pytest.raises(TypeError):
        BPETokenizer(num_merges="50")


def test_encode_before_fit_raises():
    with pytest.raises(RuntimeError):
        BPETokenizer().encode("hello")


def test_non_string_input_raises():
    with pytest.raises(TypeError):
        BPETokenizer().fit([123])
    with pytest.raises(TypeError):
        BPETokenizer().fit(42)
    tok = _fit()
    with pytest.raises(TypeError):
        tok.encode(123)
    with pytest.raises(TypeError):
        tok.decode("not a list")


def test_decode_of_encode_equals_normalised_text():
    tok = _fit()
    assert tok.decode(tok.encode("The CAT, is Learning!")) == "the cat is learning"
    assert tok.decode([]) == ""


def test_fit_accepts_single_string():
    tok = BPETokenizer(num_merges=5, min_frequency=1).fit("hello world hello")
    assert tok.merges
    assert tok.decode(tok.encode("hello world")) == "hello world"
