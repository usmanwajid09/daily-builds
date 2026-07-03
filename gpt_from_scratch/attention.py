"""Causal multi-head self-attention, forward pass only (numpy)."""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.layers import Linear


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class CausalSelfAttention:
    """Multi-head self-attention with a causal (no-peeking-ahead) mask.

    Input/output shape: (batch, seq_len, d_model).
    """

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query = Linear(d_model, d_model, rng)
        self.key = Linear(d_model, d_model, rng)
        self.value = Linear(d_model, d_model, rng)
        self.out_proj = Linear(d_model, d_model, rng)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, d_head)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, n_heads, seq_len, d_head = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq_len, n_heads * d_head)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected input shape (batch, seq_len, {self.d_model}), got {x.shape}"
            )
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)

        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(causal_mask, -np.inf, scores)

        weights = softmax(scores, axis=-1)
        attn_out = weights @ v

        merged = self._merge_heads(attn_out)
        return self.out_proj(merged)
