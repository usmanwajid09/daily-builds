# Hidden Markov Model (from scratch)

A discrete Hidden Markov Model built with numpy only: forward-backward
(with scaling, so long sequences don't underflow), Viterbi decoding,
sampling, and Baum-Welch (EM) training.

## What it does

- `HiddenMarkovModel(n_states, n_symbols, random_state)` — parameters
  stored as `start_prob_` (pi), `trans_mat_` (A), `emission_mat_` (B).
- `fit(sequences, n_iter=50, tol=1e-4)` — Baum-Welch training on a list
  of 1D int arrays (or a single 1D array). Stops early when the
  log-likelihood improves by less than `tol`. `log_likelihood_history_`
  records the total log-likelihood after each iteration.
- `log_likelihood(sequence)` — forward algorithm with scaling.
- `forward_backward(sequence)` — returns `(alpha, beta, gamma, xi)`;
  each `gamma` row sums to 1.
- `predict(sequence)` — Viterbi decoding, the most likely hidden-state
  sequence.
- `sample(n_steps, random_state=None)` — generates
  `(observations, hidden_states)`.
- `score(sequences)` — mean log-likelihood per sequence.
- `set_params(start_prob, trans_mat, emission_mat)` — set any of the
  parameters by hand (each is validated and re-normalised).

## Usage

```python
import numpy as np
from hidden_markov_model.model import HiddenMarkovModel

# True model: 2 states, 3 symbols; sample training data from it.
truth = HiddenMarkovModel(n_states=2, n_symbols=3, random_state=0)
truth.set_params(start_prob=[0.6, 0.4],
                 trans_mat=[[0.7, 0.3], [0.4, 0.6]],
                 emission_mat=[[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
sequences = [truth.sample(60, random_state=i)[0] for i in range(20)]

model = HiddenMarkovModel(n_states=2, n_symbols=3, random_state=42)
model.fit(sequences, n_iter=30)
print(model.log_likelihood(sequences[0]))
print(model.predict(sequences[0]))   # most likely hidden states
```

## Run

From the repo root:

```bash
python hidden_markov_model/demo.py
pytest tests/test_hidden_markov_model.py -q
```
