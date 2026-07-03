"""A single pre-norm transformer block: attn + FFN, each with a residual.

Forward and backward pass. Residual connections mean gradients from a
block's output flow along two paths back to its input: the identity
skip, and the branch (attn or ffn). Both contributions get summed - see
the comments in `TransformerBlock.backward`.
"""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.attention import CausalSelfAttention
from gpt_from_scratch.layers import Dropout, LayerNorm, Linear, gelu, gelu_backward


class FeedForward:
    """Two-layer MLP with GELU, expanding to 4x d_model in the hidden layer
    (the standard GPT-2 ratio)."""

    def __init__(self, d_model: int, rng: np.random.Generator, expansion: int = 4):
        hidden = d_model * expansion
        self.fc1 = Linear(d_model, hidden, rng)
        self.fc2 = Linear(hidden, d_model, rng)
        self._pre_act = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._pre_act = self.fc1.forward(x)
        h = gelu(self._pre_act)
        return self.fc2.forward(h)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._pre_act is None:
            raise RuntimeError("backward() called before forward()")
        dh = self.fc2.backward(dout)
        dpre = gelu_backward(self._pre_act, dh)
        dx = self.fc1.backward(dpre)
        return dx

    def parameters_and_grads(self):
        return self.fc1.parameters_and_grads() + self.fc2.parameters_and_grads()

    __call__ = forward


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator, dropout_p: float = 0.0):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, rng)
        self.dropout_p = dropout_p
        self.attn_dropout = Dropout(dropout_p)
        self.ffn_dropout = Dropout(dropout_p)

    def forward(self, x: np.ndarray, training: bool = True, rng: np.random.Generator = None) -> np.ndarray:
        """`training`/`rng` only matter when dropout_p > 0 (the default,
        dropout_p=0.0, makes both Dropout calls a no-op identity - so
        every pre-existing call site that only ever passes `x` keeps
        working exactly as before)."""
        a = self.attn.forward(self.ln1.forward(x))
        a = self.attn_dropout.forward(a, training=training, rng=rng)
        x2 = x + a
        f = self.ffn.forward(self.ln2.forward(x2))
        f = self.ffn_dropout.forward(f, training=training, rng=rng)
        x3 = x2 + f
        return x3

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # x3 = x2 + ffn_dropout(ffn(ln2(x2)))
        df = self.ffn_dropout.backward(dout)
        dx2_branch = self.ln2.backward(self.ffn.backward(df))
        dx2 = dout + dx2_branch  # skip connection gets the full upstream grad too

        # x2 = x + attn_dropout(attn(ln1(x)))
        da = self.attn_dropout.backward(dx2)
        dx_branch = self.ln1.backward(self.attn.backward(da))
        dx = dx2 + dx_branch

        return dx

    def forward_incremental(self, x: np.ndarray, cache):
        """Single-step (or short-chunk) forward used only during KV-cached
        generation - no dropout (always eval mode) and no backward
        support, see CausalSelfAttention.forward_incremental for the
        cache format."""
        a, new_cache = self.attn.forward_incremental(self.ln1.forward(x), cache)
        x2 = x + a
        f = self.ffn.forward(self.ln2.forward(x2))
        x3 = x2 + f
        return x3, new_cache

    def parameters_and_grads(self):
        return (
            self.ln1.parameters_and_grads()
            + self.attn.parameters_and_grads()
            + self.ln2.parameters_and_grads()
            + self.ffn.parameters_and_grads()
        )

    __call__ = forward
