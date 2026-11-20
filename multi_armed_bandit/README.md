# Multi-Armed Bandit

Three classic multi-armed bandit algorithms, implemented from scratch
with numpy only: **epsilon-greedy**, **UCB1**, and **Thompson sampling**,
plus a simulator with regret tracking.

A bandit problem: there are several slot-machine arms, each paying reward
1 with some unknown probability. The agent picks an arm each round and
learns from the reward, balancing exploration (trying uncertain arms)
against exploitation (pulling the arm that currently looks best).

## Algorithms

| Algorithm | Idea |
| --- | --- |
| `EpsilonGreedyBandit(epsilon=0.1)` | With probability `epsilon` pick a random arm, otherwise the best estimated mean. Pulls each arm once first. |
| `UCBBandit(c=2.0)` | Picks the arm maximising `value + c * sqrt(2 * ln(total_pulls) / pulls)`. Unpulled arms are tried first. |
| `ThompsonSamplingBandit()` | Keeps a Beta(1, 1) prior per arm, samples a plausible mean from each posterior, pulls the argmax. Assumes Bernoulli rewards in [0, 1]. |

Every agent has `select_arm() -> int`, `update(arm, reward) -> None`, and
the attributes `counts_` (pulls per arm), `values_` (estimated mean per
arm), `total_reward_` and `total_pulls_`. Selection is deterministic
given the `random_state` seed.

## Usage

```python
from multi_armed_bandit import EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit, run_experiment

bandit = ThompsonSamplingBandit(n_arms=4, random_state=42)
result = run_experiment(bandit, true_means=[0.1, 0.3, 0.5, 0.8], n_rounds=1000, seed=42)

print(result.total_reward)        # total reward over 1000 rounds
print(result.final_regret)        # cumulative regret vs. always pulling the best arm
print(result.chosen_arms)         # arm pulled each round
print(result.cumulative_regret)   # cumulative regret after each round
print(bandit.values_)             # estimated mean per arm
```

`run_experiment` simulates Bernoulli arms: each round the pulled arm pays
1 with its true-mean probability, else 0. Regret per round is
`best_mean - true_mean[chosen_arm]`.

## Run the demo

```bash
python multi_armed_bandit/demo.py
```

Runs all three algorithms for 1000 rounds on arms `[0.1, 0.3, 0.5, 0.8]`
and prints each one's total reward, final cumulative regret, and most-pulled arm.

## Run the tests

```bash
pytest tests/test_multi_armed_bandit.py -q
```
