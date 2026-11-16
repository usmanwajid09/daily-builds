"""Huber regression: robust linear regression with the Huber loss."""

from .model import HuberRegression, huber_loss, ols_fit

__all__ = ["HuberRegression", "huber_loss", "ols_fit"]
