import pytest

from gpt_from_scratch.bpe_tokenizer import BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox runs. the lazy dog sleeps. " * 20
)


def test_vocab_size_matches_requested():
    tok = BPETokenizer(CORPUS, vocab_size=300)
    assert tok.vocab_size == 300


def test_vocab_size_below_256_rejected():
    with pytest.raises(ValueError):
        BPETokenizer(CORPUS, vocab_size=100)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BPETokenizer("", vocab_size=256)


def test_roundtrip_on_training_corpus():
    tok = BPETokenizer(CORPUS, vocab_size=400)
    ids = tok.encode(CORPUS)
    assert tok.decode(ids) == CORPUS


def test_roundtrip_on_unseen_text():
    """The whole point of byte-level BPE: text with characters that never
    appeared in the training corpus must still encode/decode losslessly,
    unlike CharTokenizer which raises on unknown characters."""
    tok = BPETokenizer(CORPUS, vocab_size=400)
    unseen = "Héllo wörld! 🤖 new-tokenizer test — em-dash, tab\t, newline\n."
    ids = tok.encode(unseen)
    assert tok.decode(ids) == unseen


def test_compresses_repetitive_text():
    """A repetitive corpus should compress noticeably better than 1
    token/char once BPE has learned common subwords."""
    tok = BPETokenizer(CORPUS, vocab_size=400)
    ids = tok.encode(CORPUS)
    assert len(ids) < len(CORPUS) * 0.6


def test_decode_rejects_out_of_range_ids():
    tok = BPETokenizer(CORPUS, vocab_size=300)
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size + 100])


def test_merges_are_deterministic_and_reusable():
    """Two tokenizers trained on the same corpus with the same vocab_size
    must learn identical merges (no hidden randomness in training)."""
    tok1 = BPETokenizer(CORPUS, vocab_size=300)
    tok2 = BPETokenizer(CORPUS, vocab_size=300)
    assert tok1.merges == tok2.merges
    assert tok1.encode("the quick fox") == tok2.encode("the quick fox")


def test_encode_type_error_on_non_string():
    tok = BPETokenizer(CORPUS, vocab_size=300)
    with pytest.raises(TypeError):
        tok.encode(12345)
