"""Basic numpy building blocks: Linear, LayerNorm, GELU.

Milestone 2 adds a hand-written backward pass to every layer here so the
whole network can be trained with plain numpy (no autograd library). The
pattern used throughout this package: `forward(x)` caches whatever it
needs on `self`, and `backward(dout)` uses that cache to return `dx` (the
gradient w.r.t. the layer's input) while stashing parameter gradients as
`self.dW`, `self.db`, etc. `parameters_and_grads()` exposes
`(param, grad)` pairs for the optimizer.
"""
from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh approximation of GELU (same one GPT-2 uses)."""
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))


def gelu_backward(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """Analytic derivative of the tanh-approximate GELU, evaluated at the
    *pre-activation* input `x` (not the gelu output), chained with dout."""
    c = np.sqrt(2.0 / np.pi)
    inner = x + 0.044715 * np.power(x, 3)
    u = c * inner
    t = np.tanh(u)
    du_dx = c * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
    dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
    return dout * dy_dx


class Linear:
    """y = x @ W + b, with small random init."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        scale = 1.0 / np.sqrt(in_features)
        self.W = rng.normal(0, scale, size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward() called before forward()")
        x = self._x
        in_features = self.W.shape[0]
        out_features = self.W.shape[1]
        x2 = x.reshape(-1, in_features)
        dout2 = dout.reshape(-1, out_features)
        self.dW = x2.T @ dout2
        self.db = dout2.sum(axis=0)
        dx = (dout2 @ self.W.T).reshape(x.shape)
        return dx

    def parameters_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]

    __call__ = forward


class LayerNorm:
    """Standard layer norm over the last axis, with learnable gamma/beta."""

    def __init__(self, dim: int, eps: float = 1e-5):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.eps = eps
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        xhat = (x - mean) / std
        self._cache = (xhat, std)
        return xhat * self.gamma + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        xhat, std = self._cache
        n = xhat.shape[-1]
        reduce_axes = tuple(range(dout.ndim - 1))
        self.dgamma = (dout * xhat).sum(axis=reduce_axes)
        self.dbeta = dout.sum(axis=reduce_axes)

        dxhat = dout * self.gamma
        dx = (1.0 / (n * std)) * (
            n * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        return dx

    def parameters_and_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    __call__ = forward
