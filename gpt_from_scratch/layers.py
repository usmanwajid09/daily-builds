"""Basic numpy building blocks: Linear, LayerNorm, GELU."""
from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh approximation of GELU (same one GPT-2 uses)."""
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


class Linear:
    """y = x @ W + b, with small random init (no training yet - this
    milestone is forward-pass architecture only)."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        scale = 1.0 / np.sqrt(in_features)
        self.W = rng.normal(0, scale, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


class LayerNorm:
    """Standard layer norm over the last axis, with learnable-shaped
    gamma/beta (currently fixed at 1/0 - training will update them in the
    next milestone)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + self.eps)
        return normed * self.gamma + self.beta
