"""Text generation / sampling on top of TinyGPT's forward pass.

No gradients needed here. Supports temperature scaling, top-k filtering,
and nucleus (top-p) filtering for controlling how "safe vs. creative"
sampled text is, plus an optional KV-cache decode path (`use_cache=True`)
that avoids recomputing attention over the whole sequence for every new
token - see model.py's `forward_incremental` and
tests/test_kv_cache.py, which verifies the cached path produces
byte-for-byte identical logits to the full-recompute path.
"""
from __future__ import annotations

import numpy as np

from gpt_from_scratch.attention import softmax


def _filter_logits(logits: np.ndarray, top_k: int | None, top_p: float | None) -> np.ndarray:
    """Apply top-k then top-p filtering to a single (vocab_size,) logits
    vector, returning a copy with filtered-out entries set to -inf."""
    filtered = logits.copy()

    if top_k is not None and top_k < filtered.shape[0]:
        kth_value = np.partition(filtered, -top_k)[-top_k]
        filtered = np.where(filtered < kth_value, -np.inf, filtered)

    if top_p is not None:
        order = np.argsort(filtered)[::-1]  # descending
        sorted_logits = filtered[order]
        sorted_probs = softmax(sorted_logits[None, :], axis=-1)[0]
        cumulative = np.cumsum(sorted_probs)
        # keep the smallest prefix whose cumulative prob >= top_p, but
        # always keep at least the single most likely token
        cutoff = np.searchsorted(cumulative, top_p) + 1
        cutoff = max(1, min(cutoff, len(order)))
        keep_positions = order[:cutoff]
        mask = np.full_like(filtered, True, dtype=bool)
        mask[keep_positions] = False
        filtered = np.where(mask, -np.inf, filtered)

    return filtered


def _sample_next_id(next_logits, temperature, top_k, top_p, rng):
    if temperature == 0:
        return int(np.argmax(next_logits))
    scaled = next_logits / temperature
    scaled = _filter_logits(scaled, top_k, top_p)
    probs = softmax(scaled[None, :], axis=-1)[0]
    return int(rng.choice(len(probs), p=probs))


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    use_cache: bool = False,
) -> str:
    """Autoregressively sample `max_new_tokens` tokens after `prompt`.

    temperature: 0 means greedy (always pick the argmax - deterministic).
                 Higher values flatten the distribution (more random).
    top_k: if set, only sample from the k highest-probability next tokens.
    top_p: if set (nucleus sampling), only sample from the smallest set of
           highest-probability tokens whose cumulative probability is >=
           top_p. Composable with top_k (top_k filters first, then top_p
           filters what's left).
    use_cache: if True, use KV-cached incremental decoding instead of
               recomputing the full sequence on every step - much faster
               for long generations, but (in this implementation) can't
               generate past `model.max_seq_len` total tokens (prompt +
               generated), where the default False path instead silently
               slides its context window forward.
    """
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive if provided")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1] if provided")

    rng = np.random.default_rng(seed)
    ids = list(tokenizer.encode(prompt))

    if not use_cache:
        for _ in range(max_new_tokens):
            context = ids[-model.max_seq_len:]
            logits = model.forward(np.array([context]), training=False)
            next_logits = logits[0, -1]
            next_id = _sample_next_id(next_logits, temperature, top_k, top_p, rng)
            ids.append(next_id)
        return tokenizer.decode(ids)

    # --- KV-cached path ---
    if len(ids) + max_new_tokens > model.max_seq_len:
        raise ValueError(
            f"use_cache=True can't generate past max_seq_len ({model.max_seq_len}): "
            f"prompt has {len(ids)} tokens and max_new_tokens={max_new_tokens} would "
            f"reach {len(ids) + max_new_tokens}. Either shorten the prompt/max_new_tokens, "
            "or use use_cache=False (which slides its context window instead)."
        )

    prompt_ids = np.array([ids])
    logits, cache = model.forward_incremental(prompt_ids, pos_offset=0, cache=None)
    next_logits = logits[0, -1]
    pos = len(ids)

    for _ in range(max_new_tokens):
        next_id = _sample_next_id(next_logits, temperature, top_k, top_p, rng)
        ids.append(next_id)
        logits, cache = model.forward_incremental(
            np.array([[next_id]]), pos_offset=pos, cache=cache
        )
        next_logits = logits[0, -1]
        pos += 1

    return tokenizer.decode(ids)
