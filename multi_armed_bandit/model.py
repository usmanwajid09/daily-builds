"""Multi-armed bandit algorithms, implemented from scratch (numpy only).

A multi-armed bandit problem: there are ``n_arms`` slot machines ("arms"),
each paying reward 1 with some unknown probability (its true mean) and 0
otherwise. An agent repeatedly picks an arm and observes a reward, trying
to maximise total reward. The tension is exploration vs. exploitation:
pulling a poorly understood arm to learn about it, versus pulling the arm
that currently looks best.

Algorithms provided:

* :class:`EpsilonGreedyBandit` — with probability ``epsilon`` pick a random
  arm, otherwise pick the arm with the highest estimated mean.
* :class:`UCBBandit` — pick the arm maximising
  ``value + c * sqrt(2 * ln(total_pulls) / pulls_of_arm)`` (UCB1).
* :class:`ThompsonSamplingBandit` — keep a Beta(1, 1) prior per arm, sample
  a plausible mean from each posterior and pick the argmax.

Use :func:`run_experiment` to simulate any agent against Bernoulli arms
and track rewards and cumulative regret (how much reward was lost versus
always pulling the best arm).
"""

from dataclasses import dataclass

import numpy as np


def _check_n_arms(n_arms):
    if isinstance(n_arms, bool) or not isinstance(n_arms, (int, np.integer)):
        raise ValueError(f"n_arms must be an integer >= 2, got {n_arms!r}")
    if n_arms < 2:
        raise ValueError(f"n_arms must be >= 2, got {n_arms}")


def _check_reward(reward, lo=-np.inf, hi=np.inf):
    if isinstance(reward, bool) or not isinstance(
        reward, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"reward must be a finite number, got {reward!r}")
    reward = float(reward)
    if not np.isfinite(reward):
        raise ValueError(f"reward must be a finite number, got {reward!r}")
    if reward < lo or reward > hi:
        raise ValueError(f"reward must be in [{lo}, {hi}], got {reward}")
    return reward


class _BaseBandit:
    """Shared state and bookkeeping for all bandit agents."""

    def __init__(self, n_arms, random_state=None):
        _check_n_arms(n_arms)
        self.n_arms = int(n_arms)
        self.rng = np.random.default_rng(random_state)
        self.counts_ = np.zeros(self.n_arms, dtype=int)
        self.values_ = np.zeros(self.n_arms, dtype=float)
        self.total_reward_ = 0.0
        self.total_pulls_ = 0

    def _check_arm(self, arm):
        if isinstance(arm, bool) or not isinstance(arm, (int, np.integer)):
            raise ValueError(f"arm must be an integer in [0, {self.n_arms}), got {arm!r}")
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")

    def _record(self, arm, reward):
        """Update the running mean estimate for ``arm`` with ``reward``."""
        self.counts_[arm] += 1
        n = self.counts_[arm]
        # Incremental mean update: mean += (reward - mean) / n
        self.values_[arm] += (reward - self.values_[arm]) / n
        self.total_reward_ += reward
        self.total_pulls_ += 1

    def select_arm(self):
        """Return the index of the next arm to pull."""
        raise NotImplementedError

    def update(self, arm, reward):
        """Record that pulling ``arm`` returned ``reward``."""
        raise NotImplementedError


class EpsilonGreedyBandit(_BaseBandit):
    """Epsilon-greedy: explore with probability ``epsilon``, else exploit.

    Each arm is pulled once first (round-robin) so every estimate exists
    before any exploitation happens. Ties among the best estimated values
    are broken at random.
    """

    def __init__(self, n_arms, epsilon=0.1, random_state=None):
        super().__init__(n_arms, random_state)
        if isinstance(epsilon, bool) or not isinstance(
            epsilon, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"epsilon must be a number in [0, 1], got {epsilon!r}")
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon

    def select_arm(self):
        # Pull each arm once first (round-robin) so estimates exist.
        for arm in range(self.n_arms):
            if self.counts_[arm] == 0:
                return arm
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        best_value = np.max(self.values_)
        best_arms = np.flatnonzero(self.values_ == best_value)
        return int(self.rng.choice(best_arms))

    def update(self, arm, reward):
        self._check_arm(arm)
        reward = _check_reward(reward)
        self._record(int(arm), reward)


class UCBBandit(_BaseBandit):
    """UCB1: pick the arm with the highest upper confidence bound.

    ``ucb = value + c * sqrt(2 * ln(total_pulls) / counts)``. An arm that
    has never been pulled gets a bound of +inf, so every arm is tried once.
    Larger ``c`` means more exploration.
    """

    def __init__(self, n_arms, c=2.0, random_state=None):
        super().__init__(n_arms, random_state)
        if isinstance(c, bool) or not isinstance(c, (int, float, np.integer, np.floating)):
            raise ValueError(f"c must be a number > 0, got {c!r}")
        c = float(c)
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = c

    def select_arm(self):
        for arm in range(self.n_arms):
            if self.counts_[arm] == 0:
                return arm
        bonus = self.c * np.sqrt(2.0 * np.log(self.total_pulls_) / self.counts_)
        ucb = self.values_ + bonus
        best_value = np.max(ucb)
        best_arms = np.flatnonzero(ucb == best_value)
        return int(self.rng.choice(best_arms))

    def update(self, arm, reward):
        self._check_arm(arm)
        reward = _check_reward(reward)
        self._record(int(arm), reward)


class ThompsonSamplingBandit(_BaseBandit):
    """Thompson sampling with a Beta(1, 1) prior per arm.

    Assumes Bernoulli rewards in [0, 1]. Each arm keeps a count of
    successes (reward 1) and failures (reward 0); a plausible mean is
    sampled from Beta(successes + 1, failures + 1) and the arm with the
    largest sample is pulled. Arms that have done well are sampled high
    more often, but uncertain arms still get a chance.
    """

    def __init__(self, n_arms, random_state=None):
        super().__init__(n_arms, random_state)
        self.successes_ = np.zeros(self.n_arms, dtype=int)
        self.failures_ = np.zeros(self.n_arms, dtype=int)

    def select_arm(self):
        samples = self.rng.beta(self.successes_ + 1, self.failures_ + 1)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        self._check_arm(arm)
        reward = _check_reward(reward, lo=0.0, hi=1.0)
        arm = int(arm)
        if reward == 1.0:
            self.successes_[arm] += 1
        else:
            self.failures_[arm] += 1
        self._record(arm, reward)


@dataclass
class ExperimentResult:
    """Outcome of running one bandit agent against Bernoulli arms."""

    rewards: np.ndarray  # reward observed each round, shape (n_rounds,)
    chosen_arms: np.ndarray  # arm pulled each round, shape (n_rounds,)
    cumulative_regret: np.ndarray  # cumulative regret after each round
    total_reward: float
    final_regret: float


def run_experiment(bandit, true_means, n_rounds, seed=None):
    """Simulate ``bandit`` against Bernoulli arms for ``n_rounds`` rounds.

    Parameters
    ----------
    bandit : bandit agent with ``select_arm`` / ``update`` and ``n_arms``.
    true_means : sequence of arm success probabilities, each in [0, 1].
        Its length must match ``bandit.n_arms``.
    n_rounds : number of pulls, must be >= 1.
    seed : optional int seed for the environment's Bernoulli draws.

    Returns
    -------
    ExperimentResult with ``rewards``, ``chosen_arms``,
    ``cumulative_regret`` (per-round regret is ``best_mean - true_mean`` of
    the pulled arm), ``total_reward`` and ``final_regret``.
    """
    if not isinstance(n_rounds, (int, np.integer)) or isinstance(n_rounds, bool):
        raise ValueError(f"n_rounds must be an integer >= 1, got {n_rounds!r}")
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    means = np.asarray(true_means, dtype=float)
    if means.ndim != 1 or means.size < 2:
        raise ValueError(f"true_means must be a sequence of >= 2 values, got {true_means!r}")
    if not np.all(np.isfinite(means)) or np.any(means < 0.0) or np.any(means > 1.0):
        raise ValueError(f"true_means must all be in [0, 1], got {true_means!r}")
    if means.size != bandit.n_arms:
        raise ValueError(
            f"len(true_means) ({means.size}) must match bandit.n_arms ({bandit.n_arms})"
        )

    rng = np.random.default_rng(seed)
    best_mean = float(np.max(means))
    rewards = np.empty(n_rounds, dtype=float)
    chosen_arms = np.empty(n_rounds, dtype=int)
    regret = np.empty(n_rounds, dtype=float)
    cumulative = 0.0
    for t in range(n_rounds):
        arm = int(bandit.select_arm())
        reward = 1.0 if rng.random() < means[arm] else 0.0
        bandit.update(arm, reward)
        rewards[t] = reward
        chosen_arms[t] = arm
        cumulative += best_mean - means[arm]
        regret[t] = cumulative
    return ExperimentResult(
        rewards=rewards,
        chosen_arms=chosen_arms,
        cumulative_regret=regret,
        total_reward=float(rewards.sum()),
        final_regret=float(regret[-1]),
    )


# Backwards-friendly alias: run_experiment returns an ExperimentResult.
simulate = run_experiment
