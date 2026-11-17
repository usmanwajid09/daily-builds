"""Bayesian linear regression from scratch (Gaussian prior, closed-form posterior).

Model: y = X @ w + eps, with a Gaussian prior w ~ N(0, alpha^-1 I) on the
weights and Gaussian noise eps ~ N(0, beta^-1). Because the prior and the
likelihood are both Gaussian, the posterior over the weights is Gaussian
too, with a closed-form mean and covariance. Predictions therefore come
with honest uncertainty: points far from the training data get a wider
predictive distribution than points near it.
"""

from .model import BayesianRidgeRegressor

__all__ = ["BayesianRidgeRegressor"]
