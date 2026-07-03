"""Causal multi-head self-attention, forward and backward pass (numpy)."""
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
        self._cache = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, d_head)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, n_heads, seq_len, d_head = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq_len, n_heads * d_head)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected input shape (batch, seq_len, {self.d_model}), got {x.shape}"
            )
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.query.forward(x))
        k = self._split_heads(self.key.forward(x))
        v = self._split_heads(self.value.forward(x))

        scale = 1.0 / np.sqrt(self.d_head)
        raw_scores = q @ k.transpose(0, 1, 3, 2)
        scores = raw_scores * scale

        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(causal_mask, -np.inf, scores)

        weights = softmax(scores, axis=-1)
        attn_out = weights @ v

        merged = self._merge_heads(attn_out)
        out = self.out_proj.forward(merged)

        self._cache = (q, k, v, weights, causal_mask, scale)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        q, k, v, weights, causal_mask, scale = self._cache

        dmerged = self.out_proj.backward(dout)
        dattn_out = self._split_heads(dmerged)

        # attn_out = weights @ v
        dweights = dattn_out @ v.transpose(0, 1, 3, 2)
        dv = weights.transpose(0, 1, 3, 2) @ dattn_out

        # softmax backward: dscores_ij = w_ij * (dweights_ij - sum_l w_il*dweights_il)
        dscores = weights * (dweights - (weights * dweights).sum(axis=-1, keepdims=True))
        # masked positions contribute nothing (weights==0 there already zeroes
        # this out, but zero explicitly for numerical safety against -inf/0 edge cases)
        dscores = np.where(causal_mask, 0.0, dscores)

        draw = dscores * scale
        dq = draw @ k
        dk = draw.transpose(0, 1, 3, 2) @ q

        dq_merged = self._merge_heads(dq)
        dk_merged = self._merge_heads(dk)
        dv_merged = self._merge_heads(dv)

        dx_q = self.query.backward(dq_merged)
        dx_k = self.key.backward(dk_merged)
        dx_v = self.value.backward(dv_merged)
        return dx_q + dx_k + dx_v

    def parameters_and_grads(self):
        return (
            self.query.parameters_and_grads()
            + self.key.parameters_and_grads()
            + self.value.parameters_and_grads()
            + self.out_proj.parameters_and_grads()
        )

    __call__ = forward
