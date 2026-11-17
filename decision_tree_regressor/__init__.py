"""CART regression tree from scratch (numpy only).

A decision tree regressor that splits nodes by choosing the feature and
threshold that maximise variance reduction (weighted MSE decrease).
Each leaf predicts the mean of the training targets that reach it.
"""

from .model import DecisionTreeRegressor

__all__ = ["DecisionTreeRegressor"]
