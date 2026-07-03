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
