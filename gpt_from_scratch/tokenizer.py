"""Minimal character-level tokenizer.

No BPE, no external vocab files - just a sorted set of the unique
characters seen in the training corpus. Good enough for a tiny
from-scratch GPT trained on a small text file.
"""
from __future__ import annotations


class CharTokenizer:
    def __init__(self, corpus: str):
        if not corpus:
            raise ValueError("corpus must be non-empty")
        chars = sorted(set(corpus))
        self.vocab_size = len(chars)
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        unknown = set(text) - set(self._stoi)
        if unknown:
            raise ValueError(
                f"encountered characters not in vocab: {sorted(unknown)!r}"
            )
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        bad = [i for i in ids if i not in self._itos]
        if bad:
            raise ValueError(f"ids out of vocab range: {bad}")
        return "".join(self._itos[i] for i in ids)
