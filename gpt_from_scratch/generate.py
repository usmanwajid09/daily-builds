"""Text generation / sampling on top of TinyGPT's forward pass.

No gradients needed here - this only calls `model.forward`. Supports
temperature scaling and top-k filtering, the two standard knobs for
controlling how "safe vs. creative" sampled text is.
"""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.attention import softmax


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Autoregressively sample `max_new_tokens` characters after `prompt`.

    temperature: 0 means greedy (always pick the argmax - deterministic).
                 Higher values flatten the distribution (more random).
    top_k: if set, only sample from the k highest-probability next tokens.
    """
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive if provided")

    rng = np.random.default_rng(seed)
    ids = list(tokenizer.encode(prompt))

    for _ in range(max_new_tokens):
        context = ids[-model.max_seq_len:]
        logits = model.forward(np.array([context]))
        next_logits = logits[0, -1]  # (vocab_size,)

        if temperature == 0:
            next_id = int(np.argmax(next_logits))
        else:
            scaled = next_logits / temperature
            if top_k is not None and top_k < scaled.shape[0]:
                kth_value = np.partition(scaled, -top_k)[-top_k]
                scaled = np.where(scaled < kth_value, -np.inf, scaled)
            probs = softmax(scaled[None, :], axis=-1)[0]
            next_id = int(rng.choice(len(probs), p=probs))

        ids.append(next_id)

    return tokenizer.decode(ids)
