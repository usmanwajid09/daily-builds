# Adaline Classifier

An Adaptive Linear Neuron (Adaline) classifier built from scratch with numpy only.

## What it does

Trains a linear model by minimising the mean squared error between the
raw (linear) net input and the target labels, using batch gradient
descent. Features are standardised by default (zero mean, unit
variance), which matters for Adaline because the gradient step size
depends on feature scale. Training stops early when the cost stops
improving.

- Binary classification uses labels -1 / +1 internally.
- Multiclass problems are handled with a one-vs-rest ensemble.
- `predict_proba` maps the net inputs through a sigmoid and normalises
  them so each row sums to 1.

## How to run

```bash
python adaline_classifier/demo.py
```

```python
from adaline_classifier.model import AdalineClassifier

model = AdalineClassifier(learning_rate=0.05, n_iter=500)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.predict(X_test[:5]))
```

## Demo result

Two synthetic 2D clusters (100 points per class, 80/20 train/test
split), `learning_rate=0.05`, `n_iter=500`.

- Train accuracy: **0.988**
- Test accuracy: **1.000**

## Example output

```
Train accuracy: 0.988
Test accuracy:  1.000
Epochs run:     89
Final cost:     0.1471
Coefficients:   [-0.538 -0.444]
Intercept:      0.037
First 5 test predictions: [1, 1, 0, 0, 0]
First 5 true labels:      [1, 1, 0, 0, 0]
```

## Tests

11 pytest tests in `tests/test_adaline_classifier.py` — accuracy,
probabilities summing to 1, multiclass one-vs-rest, standardisation on
unscaled features, decreasing cost, and input validation.
