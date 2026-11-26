# Passive-Aggressive classifier, from scratch

A Passive-Aggressive (PA-I) online binary classifier — no scikit-learn,
just numpy. On each example it stays passive (no update) when the
prediction is correct with enough margin, and otherwise aggressively
projects the weights onto the set that classifies the example correctly.
The update step is capped by `C`, so one noisy example cannot move the
weights too far. It supports `fit` (several shuffled epochs) and
`partial_fit` (true online learning, one example at a time).

## Why

Online learners are the right tool for streams that never end — new data
arrives one example at a time and the model must adapt without retraining
from scratch. PA is the classic margin-based online algorithm, and the
`C` cap is what keeps it stable on noisy streams.

## Usage

```bash
pip install -r requirements.txt
python passive_aggressive_classifier/demo.py
```

This trains the classifier online, one example at a time, on a linearly
separable stream and reports accuracy and how many updates it made.

## Demo output (seed 42)

```
Train size: 240, test size: 60
Examples seen: 240, weight updates: 48
Test accuracy: 0.917
Weights: [4.577, 4.853], intercept: -0.633
First 5 predictions: [1, 1, 0, 0, 0]
First 5 true labels: [1, 1, 0, 0, 0]
```

## Run tests

```bash
python -m pytest tests/test_passive_aggressive_classifier.py -v
```
