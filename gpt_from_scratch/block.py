"""A single pre-norm transformer block: attn + FFN, each with a residual."""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.attention import CausalSelfAttention
from gpt_from_scratch.layers import LayerNorm, Linear, gelu


class FeedForward:
    """Two-layer MLP with GELU, expanding to 4x d_model in the hidden layer
    (the standard GPT-2 ratio)."""

    def __init__(self, d_model: int, rng: np.random.Generator, expansion: int = 4):
        hidden = d_model * expansion
        self.fc1 = Linear(d_model, hidden, rng)
        self.fc2 = Linear(hidden, d_model, rng)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, rng)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
