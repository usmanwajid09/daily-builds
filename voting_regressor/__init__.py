"""Voting regression: a Ridge, k-NN and decision-tree ensemble combined by mean, median or weighted voting (numpy only, no sklearn)."""

from .model import VotingRegressor

__all__ = ["VotingRegressor"]
