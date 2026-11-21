"""Gradient descent variants, from scratch with numpy only.

Four optimizers — plain SGD, SGD with momentum, RMSprop and Adam — sharing
one interface: ``step(params, grads)`` returns new parameters without
mutating its inputs. Plus ``minimize`` (run any optimizer on any
``(loss, grad)`` objective), ``compare_optimizers`` (race all four from the
same start), and two ready-made objectives: a quadratic and binary logistic
regression.
"""

import numpy as np

__all__ = [
    "SGD",
    "Momentum",
    "RMSprop",
    "Adam",
    "minimize",
    "compare_optimizers",
    "quadratic_loss",
    "logistic_loss",
]


def _check_lr(lr):
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"lr must be positive, got {lr!r}")


def _check_eps(eps):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")


def _check_rate(name, value):
    """Check a decay/beta/momentum-style rate lies in (0, 1)."""
    if not np.isfinite(value) or not (0 < value < 1):
        raise ValueError(f"{name} must lie in (0, 1), got {value!r}")


class _Optimizer:
    """Common base: validates shapes, lazily initialises per-shape state."""

    def _check_step_inputs(self, params, grads):
        params = np.asarray(params, dtype=float)
        grads = np.asarray(grads, dtype=float)
        if params.shape != grads.shape:
            raise ValueError(
                f"params and grads must have the same shape, "
                f"got {params.shape} and {grads.shape}"
            )
        if params.size == 0:
            raise ValueError("params must be non-empty")
        return params, grads


class SGD(_Optimizer):
    """Plain gradient descent: ``x -= lr * grad``. No state."""

    def __init__(self, lr=0.01):
        _check_lr(lr)
        self.lr = float(lr)

    def step(self, params, grads):
        params, grads = self._check_step_inputs(params, grads)
        return params - self.lr * grads


class Momentum(_Optimizer):
    """SGD with a velocity term.

    ``v = momentum * v + grad`` then ``x -= lr * v``. The velocity
    accumulates gradients, so sustained gradients accelerate the walk
    (effective step ~ ``lr / (1 - momentum)``) while oscillations cancel.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        _check_lr(lr)
        if not np.isfinite(momentum) or not (0 <= momentum < 1):
            raise ValueError(f"momentum must lie in [0, 1), got {momentum!r}")
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._v = None

    def step(self, params, grads):
        params, grads = self._check_step_inputs(params, grads)
        if self._v is None:
            self._v = np.zeros_like(params)
        self._v = self.momentum * self._v + grads
        return params - self.lr * self._v


class RMSprop(_Optimizer):
    """RMSprop: per-coordinate adaptive learning rates.

    ``cache = decay * cache + (1 - decay) * grad**2`` then
    ``x -= lr * grad / (sqrt(cache) + eps)``. Coordinates with large
    gradients get smaller steps and vice versa, which helps on
    ill-conditioned problems where plain SGD is throttled by the steepest
    direction.
    """

    def __init__(self, lr=0.01, decay=0.9, eps=1e-8):
        _check_lr(lr)
        _check_rate("decay", decay)
        _check_eps(eps)
        self.lr = float(lr)
        self.decay = float(decay)
        self.eps = float(eps)
        self._cache = None

    def step(self, params, grads):
        params, grads = self._check_step_inputs(params, grads)
        if self._cache is None:
            self._cache = np.zeros_like(params)
        self._cache = self.decay * self._cache + (1 - self.decay) * grads ** 2
        return params - self.lr * grads / (np.sqrt(self._cache) + self.eps)


class Adam(_Optimizer):
    """Adam: adaptive moments with bias correction (Kingma & Ba, 2014).

    ``m = beta1 * m + (1 - beta1) * grad`` (first moment),
    ``v = beta2 * v + (1 - beta2) * grad**2`` (second moment), both
    bias-corrected by ``1 - beta**t``, then
    ``x -= lr * m_hat / (sqrt(v_hat) + eps)``. Combines momentum with
    RMSprop-style per-coordinate scaling; the first step is roughly
    ``lr * sign(grad)``.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        _check_lr(lr)
        _check_rate("beta1", beta1)
        _check_rate("beta2", beta2)
        _check_eps(eps)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params, grads):
        params, grads = self._check_step_inputs(params, grads)
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grads
        self._v = self.beta2 * self._v + (1 - self.beta2) * grads ** 2
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def minimize(objective, x0, optimizer, n_steps=100):
    """Minimise ``objective`` with ``optimizer`` for ``n_steps`` steps.

    ``objective`` is a callable ``f(x) -> (loss, grad)`` where ``loss`` is a
    scalar and ``grad`` an array of the same shape as ``x``. Returns
    ``(x_final, loss_history)``; the history holds the loss at the start of
    each step (``n_steps`` entries). ``x0`` and the inputs are not mutated.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")
    x = np.asarray(x0, dtype=float)
    if x.size == 0:
        raise ValueError("x0 must be non-empty")
    x = x.copy()
    history = []
    for _ in range(n_steps):
        result = objective(x)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "objective must return a (loss, grad) tuple, "
                f"got {type(result).__name__}"
            )
        loss, grad = result
        loss = float(loss)
        grad = np.asarray(grad, dtype=float)
        if grad.shape != x.shape:
            raise ValueError(
                f"objective grad shape {grad.shape} does not match x shape {x.shape}"
            )
        history.append(loss)
        x = optimizer.step(x, grad)
    return x, history


def compare_optimizers(objective, x0, n_steps=200, optimizers=None):
    """Run all four optimizers on ``objective`` from the same ``x0``.

    ``optimizers`` may be a dict ``{name: optimizer}`` or a list of
    optimizers (named by class). Defaults to ``SGD(0.01)``,
    ``Momentum(0.01, 0.9)``, ``RMSprop(0.01)`` and ``Adam(0.001)``.
    Returns ``{name: {"x": x_final, "loss": final_loss, "history": [...]}}``.
    """
    if optimizers is None:
        optimizers = {
            "SGD": SGD(lr=0.01),
            "Momentum": Momentum(lr=0.01, momentum=0.9),
            "RMSprop": RMSprop(lr=0.01),
            "Adam": Adam(lr=0.001),
        }
    elif isinstance(optimizers, (list, tuple)):
        optimizers = {type(opt).__name__: opt for opt in optimizers}
    results = {}
    for name, opt in optimizers.items():
        x_final, history = minimize(objective, x0, opt, n_steps=n_steps)
        results[name] = {"x": x_final, "loss": history[-1], "history": history}
    return results


def quadratic_loss(A, b):
    """Return ``f(x) = 0.5 * x^T A x - b^T x`` as a ``(loss, grad)`` callable.

    ``A`` must be square and symmetric (positive definite for the problem
    to have a minimum); ``b`` must match its size. The optimum is at
    ``x* = A^-1 b`` with gradient ``A x - b``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A.shape}")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError(
            f"b must be a vector of length {A.shape[0]}, got shape {b.shape}"
        )
    if not np.allclose(A, A.T):
        raise ValueError("A must be symmetric")

    def objective(x):
        x = np.asarray(x, dtype=float)
        loss = 0.5 * float(x @ A @ x) - float(b @ x)
        grad = A @ x - b
        return loss, grad

    return objective


def logistic_loss(X, y):
    """Return the mean binary logistic loss as a ``(loss, grad)`` callable.

    ``X`` is the (n_samples, n_features) design matrix (append a column of
    ones yourself if you want an intercept), ``y`` the 0/1 labels.
    Uses a numerically stable sigmoid; gradient is ``X^T (p - y) / n``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"y must be a vector of length {X.shape[0]}, got shape {y.shape}"
        )
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must be binary (0/1)")

    def _sigmoid(z):
        # Stable: exp(-|z|) never overflows.
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )

    def objective(w):
        w = np.asarray(w, dtype=float)
        z = X @ w
        p = _sigmoid(z)
        eps = 1e-15
        p_clipped = np.clip(p, eps, 1 - eps)
        loss = -float(np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))
        grad = X.T @ (p - y) / X.shape[0]
        return loss, grad

    return objective
