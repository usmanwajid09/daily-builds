"""Gradient descent variants, from scratch with numpy only.

SGD, momentum, RMSprop and Adam compared on a quadratic and on binary
logistic regression.
"""

from .model import (
    SGD,
    Momentum,
    RMSprop,
    Adam,
    minimize,
    compare_optimizers,
    quadratic_loss,
    logistic_loss,
)

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
