"""KV-cache correctness: cached incremental decoding must be
mathematically identical to the full-recompute forward pass - this is
the single most important property to verify, since a subtly wrong
cache (off-by-one position, wrong mask offset, stale K/V) would still
produce plausible-looking but silently-wrong text.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpt_from_scratch.attention import CausalSelfAttention
from gpt_from_scratch.generate import generate
from gpt_from_scratch.model import TinyGPT
from gpt_from_scratch.tokenizer import CharTokenizer


def test_attention_incremental_matches_full_recompute():
    rng = np.random.default_rng(0)
    attn = CausalSelfAttention(d_model=8, n_heads=2, rng=rng)
    x = rng.normal(size=(2, 6, 8))

    out_full = attn.forward(x)

    out_prefill, cache = attn.forward_incremental(x[:, :3], cache=None)
    outs = [out_prefill]
    for t in range(3, 6):
        out_t, cache = attn.forward_incremental(x[:, t:t + 1], cache)
        outs.append(out_t)
    out_incremental = np.concatenate(outs, axis=1)

    assert np.allclose(out_full, out_incremental, atol=1e-8)


def test_attention_incremental_one_token_at_a_time_matches_full_recompute():
    """Even more granular: prefill nothing, step one token at a time from
    the very first position."""
    rng = np.random.default_rng(1)
    attn = CausalSelfAttention(d_model=8, n_heads=2, rng=rng)
    x = rng.normal(size=(1, 5, 8))

    out_full = attn.forward(x)

    cache = None
    outs = []
    for t in range(5):
        out_t, cache = attn.forward_incremental(x[:, t:t + 1], cache)
        outs.append(out_t)
    out_incremental = np.concatenate(outs, axis=1)

    assert np.allclose(out_full, out_incremental, atol=1e-8)


def test_model_forward_incremental_matches_full_forward():
    rng = np.random.default_rng(2)
    model = TinyGPT(vocab_size=10, d_model=16, n_heads=2, n_layers=2, max_seq_len=10, seed=2)
    ids = rng.integers(0, 10, size=(1, 7))

    logits_full = model.forward(ids, training=False)

    logits_pre, cache = model.forward_incremental(ids[:, :4], pos_offset=0, cache=None)
    outs = [logits_pre]
    for t in range(4, 7):
        logits_t, cache = model.forward_incremental(ids[:, t:t + 1], pos_offset=t, cache=cache)
        outs.append(logits_t)
    logits_incremental = np.concatenate(outs, axis=1)

    assert np.allclose(logits_full, logits_incremental, atol=1e-6)


def test_model_forward_incremental_rejects_exceeding_max_seq_len():
    model = TinyGPT(vocab_size=5, d_model=8, n_heads=2, n_layers=1, max_seq_len=4, seed=0)
    ids = np.array([[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        model.forward_incremental(ids, pos_offset=1, cache=None)  # 1+4 > 4


def test_generate_cached_matches_uncached_for_greedy_decoding():
    """The end-to-end check: use_cache=True and use_cache=False must
    produce IDENTICAL text under greedy (temperature=0) decoding, since
    both are deterministic and mathematically equivalent."""
    tok = CharTokenizer("abcdefghij")
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=2, max_seq_len=20, seed=0)

    out_nocache = generate(model, tok, prompt="a", max_new_tokens=10, temperature=0, use_cache=False)
    out_cache = generate(model, tok, prompt="a", max_new_tokens=10, temperature=0, use_cache=True)
    assert out_nocache == out_cache


def test_generate_cached_rejects_exceeding_context_window():
    tok = CharTokenizer("abcdefghij")
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=10, seed=0)
    with pytest.raises(ValueError):
        generate(model, tok, prompt="a", max_new_tokens=15, use_cache=True)


# ---------- top-p sampling ----------

def test_top_p_rejects_out_of_range():
    tok = CharTokenizer("abcdef")
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=16, seed=0)
    with pytest.raises(ValueError):
        generate(model, tok, prompt="a", max_new_tokens=5, top_p=0.0)
    with pytest.raises(ValueError):
        generate(model, tok, prompt="a", max_new_tokens=5, top_p=1.5)


def test_top_p_one_is_unrestricted_and_reproducible():
    """top_p=1.0 keeps the whole distribution - should behave like plain
    temperature sampling (still reproducible given the same seed)."""
    tok = CharTokenizer("abcdef")
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=16, seed=0)
    out1 = generate(model, tok, prompt="a", max_new_tokens=8, temperature=1.0, top_p=1.0, seed=3)
    out2 = generate(model, tok, prompt="a", max_new_tokens=8, temperature=1.0, top_p=1.0, seed=3)
    assert out1 == out2


def test_top_p_small_value_still_produces_valid_output():
    tok = CharTokenizer("abcdef")
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=16, seed=0)
    out = generate(model, tok, prompt="a", max_new_tokens=10, temperature=1.0, top_p=0.1, seed=0)
    assert len(out) == 11
    assert set(out) <= set("abcdef")


def test_top_p_composes_with_top_k():
    tok = CharTokenizer("abcdef")
    model = TinyGPT(vocab_size=tok.vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=16, seed=0)
    out = generate(model, tok, prompt="a", max_new_tokens=10, temperature=1.0, top_k=4, top_p=0.9, seed=0)
    assert len(out) == 11
