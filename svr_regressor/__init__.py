"""Support Vector Regression (SVR) from scratch, implemented with NumPy only.

Trains a linear regressor by minimising the epsilon-insensitive loss with
L2 regularisation, optimised by stochastic gradient descent on the primal.
"""

from .model import SVRRegressor

__all__ = ["SVRRegressor"]
