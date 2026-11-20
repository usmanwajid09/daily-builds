"""Weather example: learn a 2-state HMM (Rainy / Sunny) from sampled
observations (walk / shop / clean), then decode one sequence with
Viterbi and compare against the true hidden states.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from hidden_markov_model.model import HiddenMarkovModel

# True model, in the style of Rabiner's classic weather example but
# with well-separated emissions so the states are actually recoverable.
# States: 0 = Rainy, 1 = Sunny. Symbols: 0 = walk, 1 = shop, 2 = clean.
TRUE_START = np.array([0.6, 0.4])
TRUE_TRANS = np.array([[0.8, 0.2],
                       [0.2, 0.8]])
TRUE_EMIT = np.array([[0.1, 0.1, 0.8],   # Rainy: mostly clean
                      [0.8, 0.1, 0.1]])  # Sunny: mostly walk


def best_alignment_accuracy(decoded, true):
    """Accuracy under the better of the two state-label alignments
    (Baum-Welch may swap the labels of the learned states)."""
    return max(float(np.mean(decoded == true)),
               float(np.mean((1 - decoded) == true)))


def main():
    rng = np.random.default_rng(0)
    truth = HiddenMarkovModel(n_states=2, n_symbols=3, random_state=0)
    truth.set_params(start_prob=TRUE_START, trans_mat=TRUE_TRANS,
                     emission_mat=TRUE_EMIT)
    sequences, true_states = [], []
    for _ in range(30):
        obs, states = truth.sample(80, random_state=int(rng.integers(1 << 30)))
        sequences.append(obs)
        true_states.append(states)

    # Log-likelihood of a random initialisation, for the before/after.
    model = HiddenMarkovModel(n_states=2, n_symbols=3, random_state=42)
    model._fitted = True  # random init params are usable for scoring
    ll_before = model.score(sequences)
    model._fitted = False

    # Baum-Welch from 3 random restarts; keep the best final likelihood.
    best = None
    for restart in range(3):
        candidate = HiddenMarkovModel(n_states=2, n_symbols=3,
                                      random_state=42 + restart)
        candidate.fit(sequences, n_iter=50)
        ll = candidate.score(sequences)
        if best is None or ll > best[0]:
            best = (ll, candidate)
    ll_after, model = best

    print(f"Mean log-likelihood per sequence before training: {ll_before:.2f}")
    print(f"Mean log-likelihood per sequence after training:  {ll_after:.2f}")
    print(f"Log-likelihood improved by {ll_after - ll_before:.2f}")
    print("Learned transition matrix (true: [[0.8, 0.2], [0.2, 0.8]]):")
    print(np.round(model.trans_mat_, 3))
    print("Learned emission matrix (true: [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]):")
    print(np.round(model.emission_mat_, 3))

    accuracies = [best_alignment_accuracy(model.predict(seq), true)
                  for seq, true in zip(sequences, true_states)]
    print(f"Mean Viterbi accuracy over 30 sequences: {np.mean(accuracies):.3f}")
    decoded = model.predict(sequences[0])
    print(f"Viterbi accuracy on sequence 0: {accuracies[0]:.3f}")
    print(f"True states (first 30):    {true_states[0][:30].tolist()}")
    print(f"Decoded states (first 30): {decoded[:30].tolist()}")


if __name__ == "__main__":
    main()
