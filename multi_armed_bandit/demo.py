"""Compare epsilon-greedy, UCB1 and Thompson sampling on Bernoulli arms.

Runs each algorithm for 1000 rounds against four arms with true success
probabilities [0.1, 0.3, 0.5, 0.8] and prints total reward, final
cumulative regret, and which arm each algorithm pulled most often.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from multi_armed_bandit import (
    EpsilonGreedyBandit,
    ThompsonSamplingBandit,
    UCBBandit,
    run_experiment,
)

TRUE_MEANS = [0.1, 0.3, 0.5, 0.8]
N_ROUNDS = 1000
SEED = 42


def main():
    agents = [
        ("Epsilon-greedy (eps=0.1)", EpsilonGreedyBandit(4, epsilon=0.1, random_state=SEED)),
        ("UCB1 (c=2.0)", UCBBandit(4, c=2.0, random_state=SEED)),
        ("Thompson sampling", ThompsonSamplingBandit(4, random_state=SEED)),
    ]
    print(f"True arm means: {TRUE_MEANS}  (best arm: 3, mean 0.8)")
    print(f"Rounds: {N_ROUNDS}, seed: {SEED}\n")
    for name, agent in agents:
        result = run_experiment(agent, TRUE_MEANS, N_ROUNDS, seed=SEED)
        counts = np.bincount(result.chosen_arms, minlength=4)
        most_pulled = int(np.argmax(counts))
        print(f"{name}")
        print(f"  Total reward:          {result.total_reward:.0f} / {N_ROUNDS}")
        print(f"  Final cumulative regret: {result.final_regret:.1f}")
        print(f"  Pulls per arm:         {counts.tolist()}")
        print(f"  Most-pulled arm:       {most_pulled} ({counts[most_pulled]} pulls)")
        print(f"  Estimated values:      {[round(v, 3) for v in agent.values_]}")


if __name__ == "__main__":
    main()
