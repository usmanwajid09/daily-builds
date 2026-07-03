import numpy as np
import pytest

from gpt_from_scratch.attention import CausalSelfAttention, softmax


def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(0).normal(size=(3, 5))
    out = softmax(x, axis=-1)
    assert np.allclose(out.sum(axis=-1), 1.0)


def test_attention_output_shape():
    rng = np.random.default_rng(0)
    attn = CausalSelfAttention(d_model=16, n_heads=4, rng=rng)
    x = rng.normal(size=(2, 5, 16))
    out = attn(x)
    assert out.shape == (2, 5, 16)


def test_rejects_d_model_not_divisible_by_heads():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        CausalSelfAttention(d_model=10, n_heads=3, rng=rng)


def test_causal_mask_blocks_future_tokens():
    """Changing a future token's values must not change an earlier
    position's output - that's what the causal mask guarantees."""
    rng = np.random.default_rng(0)
    attn = CausalSelfAttention(d_model=8, n_heads=2, rng=rng)
    x = rng.normal(size=(1, 4, 8))
    out_before = attn(x)

    x_modified = x.copy()
    x_modified[0, -1, :] = 999.0  # perturb only the LAST position
    out_after = attn(x_modified)

    # All positions except the last must be unaffected by that change.
    assert np.allclose(out_before[0, :-1], out_after[0, :-1], atol=1e-8)
