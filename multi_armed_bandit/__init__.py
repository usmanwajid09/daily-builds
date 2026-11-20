"""Multi-armed bandit algorithms, implemented from scratch."""

from multi_armed_bandit.model import (
    EpsilonGreedyBandit,
    ExperimentResult,
    ThompsonSamplingBandit,
    UCBBandit,
    run_experiment,
    simulate,
)

__all__ = [
    "EpsilonGreedyBandit",
    "UCBBandit",
    "ThompsonSamplingBandit",
    "run_experiment",
    "simulate",
    "ExperimentResult",
]
