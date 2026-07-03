"""Tiny GPT: token + positional embeddings, stacked transformer blocks,
final layer norm, and a linear head projecting back to vocab logits.

Milestone 2 added `backward()`: given dL/dlogits, it walks the whole
network in reverse (head -> final layernorm -> blocks in reverse order ->
embeddings) and accumulates gradients into every parameter, ready for an
optimizer step.

This upgrade adds three opt-in, backward-compatible extras (all default
to "off", so `TinyGPT(vocab_size=...)` with no other args behaves
identically to the milestone-2 model, and every milestone-1/2 test still
passes unmodified):
  - `dropout_p`: residual dropout inside every block (see block.py).
  - `tie_weights`: share the output head's weight matrix with the token
    embedding (see layers.TiedHead) instead of a separate Linear.
  - `forward_incremental` / cached generation: see generate.py's
    `use_cache` path, which avoids recomputing the whole sequence for
    every new token.
"""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.block import TransformerBlock
from gpt_from_scratch.layers import Dropout, LayerNorm, Linear, TiedHead


class TinyGPT:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
        seed: int = 0,
        dropout_p: float = 0.0,
        tie_weights: bool = False,
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
        self.tie_weights = tie_weights
        rng = np.random.default_rng(seed)

        self.token_emb = rng.normal(0, 0.02, size=(vocab_size, d_model))
        self.pos_emb = rng.normal(0, 0.02, size=(max_seq_len, d_model))
        self.d_token_emb = np.zeros_like(self.token_emb)
        self.d_pos_emb = np.zeros_like(self.pos_emb)

        self.blocks = [
            TransformerBlock(d_model, n_heads, rng, dropout_p=dropout_p)
            for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)
        self.emb_dropout = Dropout(dropout_p)

        if tie_weights:
            self.head = TiedHead(self.token_emb, vocab_size)
        else:
            self.head = Linear(d_model, vocab_size, rng)

        self._token_ids = None

    def forward(self, token_ids: np.ndarray, training: bool = True, rng: np.random.Generator = None) -> np.ndarray:
        """token_ids: (batch, seq_len) int array -> logits (batch, seq_len, vocab_size).

        `training`/`rng` only matter if dropout_p > 0 - with the default
        dropout_p=0.0 every Dropout call below is a no-op, so old call
        sites (`model(token_ids)`, `model.forward(token_ids)`) are
        unaffected."""
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be a 2D (batch, seq_len) array")
        batch, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        if batch == 0 or seq_len == 0:
            raise ValueError("token_ids must not be empty (batch and seq_len must both be > 0)")
        if token_ids.min() < 0 or token_ids.max() >= self.vocab_size:
            raise ValueError("token_ids contains an id outside [0, vocab_size)")

        self._token_ids = token_ids
        x = self.token_emb[token_ids] + self.pos_emb[:seq_len]
        x = self.emb_dropout.forward(x, training=training, rng=rng)

        for block in self.blocks:
            x = block.forward(x, training=training, rng=rng)

        x = self.ln_final.forward(x)
        logits = self.head.forward(x)
        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        """dlogits: dL/dlogits, shape (batch, seq_len, vocab_size). Populates
        gradients on every parameter in the model (see parameters_and_grads);
        there's no further "input" to return a gradient for."""
        if self._token_ids is None:
            raise RuntimeError("backward() called before forward()")

        dx = self.head.backward(dlogits)
        dx = self.ln_final.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        dx = self.emb_dropout.backward(dx)

        token_ids = self._token_ids
        seq_len = token_ids.shape[1]

        self.d_token_emb = np.zeros_like(self.token_emb)
        np.add.at(self.d_token_emb, token_ids, dx)
        if self.tie_weights:
            # token_emb also received gradient through the output head
            # (it's the same array, used twice) - sum both contributions.
            self.d_token_emb = self.d_token_emb + self.head.d_token_emb_contrib

        self.d_pos_emb = np.zeros_like(self.pos_emb)
        self.d_pos_emb[:seq_len] = dx.sum(axis=0)

    def forward_incremental(self, token_ids: np.ndarray, pos_offset: int, cache):
        """Process a short chunk of new tokens (often just one) given the
        KV-cache from all previously-processed tokens, instead of
        recomputing attention over the whole sequence from scratch. No
        dropout (always eval mode), no backward support - generation only.

        token_ids: (batch, chunk_len). pos_offset: how many tokens have
        already been fed to the model (so positional embeddings line up).
        cache: None (first call) or a list of per-block (k, v) caches
        from a previous call. Returns (logits, new_cache).
        """
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be a 2D (batch, chunk_len) array")
        batch, chunk_len = token_ids.shape
        if pos_offset + chunk_len > self.max_seq_len:
            raise ValueError(
                f"pos_offset + chunk_len ({pos_offset + chunk_len}) exceeds "
                f"max_seq_len {self.max_seq_len} - KV-cache generation in this "
                "implementation doesn't support sliding past the context window"
            )
        if token_ids.min() < 0 or token_ids.max() >= self.vocab_size:
            raise ValueError("token_ids contains an id outside [0, vocab_size)")

        x = self.token_emb[token_ids] + self.pos_emb[pos_offset:pos_offset + chunk_len]

        if cache is None:
            cache = [None] * len(self.blocks)
        new_cache = []
        for block, block_cache in zip(self.blocks, cache):
            x, updated = block.forward_incremental(x, block_cache)
            new_cache.append(updated)

        x = self.ln_final.forward(x)
        logits = self.head.forward(x)
        return logits, new_cache

    def parameters_and_grads(self):
        params = [(self.token_emb, self.d_token_emb), (self.pos_emb, self.d_pos_emb)]
        for block in self.blocks:
            params += block.parameters_and_grads()
        params += self.ln_final.parameters_and_grads()
        params += self.head.parameters_and_grads()
        return params

    __call__ = forward
