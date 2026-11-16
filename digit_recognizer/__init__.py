"""Digit recogniser from scratch (numpy only)."""

from digit_recognizer.digits import (
    TEMPLATES,
    TEMPLATE_SHAPE,
    accuracy,
    add_noise,
    distances,
    recognize,
    template,
)

__all__ = [
    "TEMPLATES",
    "TEMPLATE_SHAPE",
    "accuracy",
    "add_noise",
    "distances",
    "recognize",
    "template",
]
