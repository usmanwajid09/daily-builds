"""Stacking regressor: ridge + k-NN + decision tree, combined by a ridge meta-learner."""

from .model import StackingRegressor

__all__ = ["StackingRegressor"]
