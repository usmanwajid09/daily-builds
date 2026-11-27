# RNN from Scratch

A vanilla recurrent neural network (RNN) sequence classifier built from
scratch with **numpy only** — no deep-learning framework.

## What it does

Given a sequence of inputs (one vector per time step), the model predicts
a single class label for the whole sequence. The demo task is **parity**:
given a sequence of 6 bits, predict whether the number of 1s is odd or
even. No single time step carries the answer, so the network has to carry
information forward through its hidden state.

## How it works

- **Recurrence (tanh hidden state):**
  `h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)` — the hidden state is the
  network's memory, updated one time step at a time.
- **Output:** the class is predicted from the *final* hidden state only:
  `logits = Why @ h_T + by`, then a softmax turns the logits into
  probabilities.
- **Training — full backpropagation through time (BPTT):** the
  cross-entropy loss gradient is propagated backwards through every time
  step, using `tanh'(z) = 1 - h_t^2` at each step, accumulating gradients
  for `Wxh`, `Whh`, `bh`, `Why`, `by`.
- **Optimiser:** plain mini-batch stochastic gradient descent (batches
  of 32, one update per batch), with gradients clipped to `[-5, 5]` so
  the recurrent gradients cannot explode.
- **Initialisation:** small random weights (Xavier-ish scale
  `1/sqrt(fan_in)`), with a fixed seed for reproducibility.

## How to run

```bash
cd /home/hatch/workspace/daily-builds
python rnn_from_scratch/demo.py
python -m pytest tests/test_rnn_from_scratch.py -q
```

## Example output

```
Train set: (400, 6, 2), test set: (100, 6, 2) (0=even parity, 1=odd parity)
Initial loss: 0.7349
Final loss:   0.0095
Loss decreased: True
Train accuracy: 1.000
Test accuracy:  1.000
Fresh sequences (true [1, 0, 0, 0, 0]): predicted [1, 0, 0, 0, 0]
Confidence on first fresh sample: 0.991 (probabilities sum to 1.000)
```

## API

```python
from rnn_from_scratch import RNNClassifier
from rnn_from_scratch.model import make_parity_dataset

X, y = make_parity_dataset(400, seq_len=6, seed=42)
clf = RNNClassifier(hidden_size=32, learning_rate=0.1, epochs=150, seed=42)
clf.fit(X, y)                 # returns self; stores clf.loss_history_
clf.predict(X)                # class labels, shape (n_samples,)
clf.predict_proba(X)          # probabilities, rows sum to 1
clf.score(X, y)               # accuracy in [0, 1]
```

`fit` expects `X` of shape `(n_samples, seq_len, input_size)` and integer
labels `y` of shape `(n_samples,)` with at least 2 classes. Invalid
inputs raise `ValueError` with a clear message.
