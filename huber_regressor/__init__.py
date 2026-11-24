"""Robust linear regression with the Huber loss, fitted by IRLS.

The Huber loss is quadratic for small residuals and linear for large
ones, so gross outliers are down-weighted instead of dragging the fit
around the way they do with ordinary least squares.
"""

from .model import HuberRegressor

__all__ = ["HuberRegressor"]
