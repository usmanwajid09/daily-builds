"""Quantile regression by gradient descent on the pinball loss (numpy only)."""

from .model import QuantileRegressor, pinball_loss

__all__ = ["QuantileRegressor", "pinball_loss"]
