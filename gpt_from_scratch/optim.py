"""Adam optimizer, written against the `parameters_and_grads()` protocol
every layer in this package implements: a list of (param_array,
grad_array) pairs, where param_array is the *actual* array object owned
by the model (not a copy) so in-place updates here are visible to it.
"""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m = None
        self._v = None

    def step(self, params_and_grads) -> None:
        """params_and_grads: list of (param, grad) numpy array pairs, in a
        *stable order* across calls (state is keyed by list position)."""
        params_and_grads = list(params_and_grads)
        if self._m is None:
            self._m = [np.zeros_like(p) for p, _ in params_and_grads]
            self._v = [np.zeros_like(p) for p, _ in params_and_grads]
        elif len(self._m) != len(params_and_grads):
            raise ValueError(
                "number of parameters changed between steps - Adam state is "
                "keyed by position and can't be reused across model shapes"
            )

        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t

        for i, (p, g) in enumerate(params_and_grads):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bias_correction1
            v_hat = self._v[i] / bias_correction2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm(params_and_grads, max_norm: float) -> float:
    """Rescale all gradients in-place so their combined L2 norm is at most
    `max_norm` (the standard "global norm" gradient clipping used to stop
    a few exploding gradients - e.g. from a bad batch - from blowing up
    the whole update). Returns the pre-clipping norm, mostly useful for
    logging/debugging.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    total_sq = 0.0
    for _, g in params_and_grads:
        total_sq += float(np.sum(g * g))
    total_norm = float(np.sqrt(total_sq))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for _, g in params_and_grads:
            g *= scale
    return total_norm


def cosine_lr_with_warmup(
    step: int,
    max_lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay from `max_lr`
    down to `max_lr * min_lr_ratio` by `max_steps`, then held flat at the
    floor. The standard GPT-family schedule: too-high a constant LR tends
    to destabilize early training before the model has any sense of
    scale, and decaying at the end lets it settle into a sharper minimum.
    """
    if warmup_steps < 0 or max_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and max_steps must be > 0")
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return max_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    floor = max_lr * min_lr_ratio
    return floor + (max_lr - floor) * cosine
