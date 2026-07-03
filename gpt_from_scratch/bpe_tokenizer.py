"""Byte-level BPE tokenizer (the same family of algorithm GPT-2 uses),
implemented from scratch - no `tiktoken`, no `transformers`.

Why byte-level: operating on raw UTF-8 bytes (0-255) as the base
vocabulary, rather than characters, means encode() can NEVER hit an
"unknown symbol" - any string, in any language, with any emoji or typo,
decomposes into some sequence of the 256 base byte tokens even before any
merges are learned. That's strictly more general than the milestone-1/2
`CharTokenizer`, which raises on any character outside the training
corpus.

Algorithm (standard BPE, e.g. Sennrich et al. 2015 / the "minBPE" style
GPT-2 uses): start from raw bytes, repeatedly find the most frequent
adjacent pair of tokens and merge it into a new token, until the target
vocab size is reached. `encode()` replays those learned merges in the
order they were learned; `decode()` just concatenates each token's raw
bytes back together and decodes as UTF-8.
"""
from __future__ import annotations


def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    """Byte-level BPE, trained on a single corpus string.

    vocab_size must be >= 256 (the 256 base byte tokens). Training stops
    early if the corpus runs out of repeated pairs before hitting the
    target vocab size (e.g. a very short or very diverse corpus).
    """

    def __init__(self, corpus: str, vocab_size: int = 512):
        if not corpus:
            raise ValueError("corpus must be non-empty")
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the base byte vocab)")

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}  # pair -> new_id, in learned order

        ids = list(corpus.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            pair_counts = _get_pair_counts(ids)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # merging a pair that only occurs once buys nothing
            new_id = 256 + i
            ids = _merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode() expects a str")
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = _get_pair_counts(ids)
            # apply the earliest-learned applicable merge first - BPE merge
            # order matters, so we can't just pick the most frequent pair
            # in `ids` here the way training does.
            candidate = min(
                pair_counts, key=lambda p: self.merges.get(p, float("inf"))
            )
            if candidate not in self.merges:
                break
            ids = _merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        bad = [i for i in ids if i not in self.vocab]
        if bad:
            raise ValueError(f"ids out of vocab range: {bad}")
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
