import numpy as np
import pytest

from gpt_from_scratch.generate import generate
from gpt_from_scratch.model import TinyGPT
from gpt_from_scratch.tokenizer import CharTokenizer


def _tiny_model_and_tok():
    tok = CharTokenizer("abcdef")
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=16, seed=0)
    return model, tok


def test_generate_length_and_vocab():
    model, tok = _tiny_model_and_tok()
    out = generate(model, tok, prompt="a", max_new_tokens=10, temperature=1.0, top_k=3, seed=0)
    assert len(out) == 11  # prompt + max_new_tokens
    assert set(out) <= set("abcdef")


def test_generate_temperature_zero_is_deterministic_argmax():
    model, tok = _tiny_model_and_tok()
    out1 = generate(model, tok, prompt="a", max_new_tokens=8, temperature=0)
    out2 = generate(model, tok, prompt="a", max_new_tokens=8, temperature=0)
    assert out1 == out2


def test_generate_same_seed_is_reproducible_when_sampling():
    model, tok = _tiny_model_and_tok()
    out1 = generate(model, tok, prompt="a", max_new_tokens=8, temperature=1.0, seed=42)
    out2 = generate(model, tok, prompt="a", max_new_tokens=8, temperature=1.0, seed=42)
    assert out1 == out2


def test_generate_rejects_empty_prompt():
    model, tok = _tiny_model_and_tok()
    with pytest.raises(ValueError):
        generate(model, tok, prompt="", max_new_tokens=5)


def test_generate_rejects_negative_temperature():
    model, tok = _tiny_model_and_tok()
    with pytest.raises(ValueError):
        generate(model, tok, prompt="a", max_new_tokens=5, temperature=-1)


def test_generate_rejects_bad_top_k():
    model, tok = _tiny_model_and_tok()
    with pytest.raises(ValueError):
        generate(model, tok, prompt="a", max_new_tokens=5, top_k=0)


def test_generate_respects_context_window_longer_than_prompt():
    """max_seq_len=16 but we generate well past that - generate() must
    truncate context to the last max_seq_len tokens rather than crashing."""
    model, tok = _tiny_model_and_tok()
    out = generate(model, tok, prompt="a", max_new_tokens=40, temperature=1.0, top_k=2, seed=1)
    assert len(out) == 41
