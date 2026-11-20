import numpy as np
import pytest

from multi_armed_bandit import (
    EpsilonGreedyBandit,
    ThompsonSamplingBandit,
    UCBBandit,
    run_experiment,
)

MEANS = [0.1, 0.5, 0.8]
BEST = 2
AGENTS = [
    lambda seed: EpsilonGreedyBandit(3, epsilon=0.1, random_state=seed),
    lambda seed: UCBBandit(3, c=2.0, random_state=seed),
    lambda seed: ThompsonSamplingBandit(3, random_state=seed),
]


@pytest.mark.parametrize("make_agent", AGENTS)
def test_pulls_every_arm_at_least_once(make_agent):
    agent = make_agent(0)
    run_experiment(agent, MEANS, 100, seed=0)
    assert np.all(agent.counts_ > 0)


@pytest.mark.parametrize("make_agent", AGENTS)
def test_best_arm_pulled_most_often(make_agent):
    agent = make_agent(1)
    result = run_experiment(agent, MEANS, 600, seed=1)
    counts = np.bincount(result.chosen_arms, minlength=3)
    assert counts[BEST] == counts.max()
    assert counts[BEST] / 600 > 0.4


@pytest.mark.parametrize("make_agent", AGENTS)
def test_final_regret_below_bound(make_agent):
    agent = make_agent(2)
    result = run_experiment(agent, MEANS, 1000, seed=2)
    assert result.final_regret < 200


def test_thompson_not_much_worse_than_epsilon_greedy():
    eps = run_experiment(
        EpsilonGreedyBandit(3, epsilon=0.1, random_state=3), MEANS, 1000, seed=3
    )
    ts = run_experiment(
        ThompsonSamplingBandit(3, random_state=3), MEANS, 1000, seed=3
    )
    # Thompson sampling is expected to beat epsilon-greedy; allow slack.
    assert ts.final_regret <= eps.final_regret + 50


def test_select_arm_returns_valid_arm_ids():
    for make_agent in AGENTS:
        agent = make_agent(4)
        for _ in range(50):
            arm = agent.select_arm()
            assert isinstance(arm, int)
            assert 0 <= arm < 3
            agent.update(arm, 1.0)


def test_update_rejects_bad_arm_and_reward():
    agent = EpsilonGreedyBandit(3, random_state=0)
    with pytest.raises(ValueError):
        agent.update(-1, 1.0)
    with pytest.raises(ValueError):
        agent.update(3, 1.0)
    with pytest.raises(ValueError):
        agent.update(0, float("nan"))
    with pytest.raises(ValueError):
        agent.update(0, float("inf"))
    # Thompson sampling only accepts Bernoulli rewards in [0, 1].
    ts = ThompsonSamplingBandit(3, random_state=0)
    with pytest.raises(ValueError):
        ts.update(0, 1.5)
    with pytest.raises(ValueError):
        ts.update(0, -0.5)


def test_counts_sum_equals_total_pulls():
    agent = UCBBandit(3, random_state=5)
    run_experiment(agent, MEANS, 200, seed=5)
    assert agent.counts_.sum() == agent.total_pulls_ == 200
    assert agent.total_reward_ == pytest.approx(
        float(np.sum(agent.values_ * agent.counts_)), abs=1e-6
    )


def test_values_converge_near_true_mean_for_best_arm():
    agent = ThompsonSamplingBandit(3, random_state=6)
    run_experiment(agent, MEANS, 1000, seed=6)
    assert agent.values_[BEST] == pytest.approx(0.8, abs=0.15)


def test_epsilon_greedy_pulls_each_arm_once_first():
    agent = EpsilonGreedyBandit(3, epsilon=0.0, random_state=0)
    first_pulls = []
    for _ in range(3):
        arm = agent.select_arm()
        first_pulls.append(arm)
        agent.update(arm, 1.0)
    assert first_pulls == [0, 1, 2]


def test_ucb_pulls_each_arm_once_first():
    agent = UCBBandit(3, random_state=0)
    first_pulls = []
    for _ in range(3):
        arm = agent.select_arm()
        first_pulls.append(arm)
        agent.update(arm, 1.0)
    assert first_pulls == [0, 1, 2]


def test_cumulative_regret_is_monotone_and_consistent():
    agent = EpsilonGreedyBandit(3, epsilon=0.1, random_state=7)
    result = run_experiment(agent, MEANS, 500, seed=7)
    assert len(result.rewards) == len(result.chosen_arms) == 500
    assert len(result.cumulative_regret) == 500
    assert np.all(np.diff(result.cumulative_regret) >= 0)
    assert result.cumulative_regret[-1] == pytest.approx(result.final_regret)
    assert result.rewards.sum() == pytest.approx(result.total_reward)


def test_invalid_n_arms():
    with pytest.raises(ValueError):
        EpsilonGreedyBandit(1)
    with pytest.raises(ValueError):
        UCBBandit(0)
    with pytest.raises(ValueError):
        ThompsonSamplingBandit(1)


def test_invalid_epsilon():
    with pytest.raises(ValueError):
        EpsilonGreedyBandit(3, epsilon=-0.1)
    with pytest.raises(ValueError):
        EpsilonGreedyBandit(3, epsilon=1.1)


def test_invalid_ucb_c():
    with pytest.raises(ValueError):
        UCBBandit(3, c=0.0)
    with pytest.raises(ValueError):
        UCBBandit(3, c=-1.0)


def test_run_experiment_validates_inputs():
    agent = EpsilonGreedyBandit(3, random_state=0)
    with pytest.raises(ValueError):
        run_experiment(agent, [0.1, 1.5, 0.8], 100, seed=0)
    with pytest.raises(ValueError):
        run_experiment(agent, [-0.1, 0.5, 0.8], 100, seed=0)
    with pytest.raises(ValueError):
        run_experiment(agent, MEANS, 0, seed=0)
    with pytest.raises(ValueError):
        run_experiment(agent, [0.1, 0.5], 100, seed=0)  # wrong arm count


def test_same_seed_is_reproducible():
    r1 = run_experiment(EpsilonGreedyBandit(3, random_state=8), MEANS, 300, seed=8)
    r2 = run_experiment(EpsilonGreedyBandit(3, random_state=8), MEANS, 300, seed=8)
    assert np.array_equal(r1.chosen_arms, r2.chosen_arms)
    assert np.array_equal(r1.rewards, r2.rewards)
