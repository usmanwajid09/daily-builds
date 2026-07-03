"""Tiny GPT: token + positional embeddings, stacked transformer blocks,
final layer norm, and a linear head projecting back to vocab logits.

Forward pass only in this milestone - no training yet (see REVIEW.md).
"""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.block import TransformerBlock
from gpt_from_scratch.layers import LayerNorm, Linear


class TinyGPT:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
        seed: int = 0,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        rng = np.random.default_rng(seed)

        self.token_emb = rng.normal(0, 0.02, size=(vocab_size, d_model))
        self.pos_emb = rng.normal(0, 0.02, size=(max_seq_len, d_model))

        self.blocks = [
            TransformerBlock(d_model, n_heads, rng) for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (batch, seq_len) int array -> logits (batch, seq_len, vocab_size)."""
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be a 2D (batch, seq_len) array")
        batch, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        if token_ids.min() < 0 or token_ids.max() >= self.vocab_size:
            raise ValueError("token_ids contains an id outside [0, vocab_size)")

        x = self.token_emb[token_ids] + self.pos_emb[:seq_len]

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.head(x)
        return logits
