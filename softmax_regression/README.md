# Softmax regression, from scratch

Multinomial logistic regression — the multi-class version of logistic
regression — implemented with numpy only, no scikit-learn. A linear
score is computed for every class, the softmax turns those scores into
a probability distribution, and the weights are trained by minimising
cross-entropy loss with batch gradient descent. Features are
standardised inside `fit` so gradient descent converges regardless of
the input scale, and the softmax subtracts the row max first so large
scores never overflow.

## Why

Binary logistic regression stops at two classes. Softmax regression is
the natural extension: one weight vector per class, probabilities that
sum to 1, and a loss (cross-entropy) that directly rewards confident
correct predictions. It is the simplest honest multi-class classifier
and the output layer of most neural networks, so understanding it from
scratch pays off everywhere else in ML.

## Usage

From the repo root:

```bash
python softmax_regression/demo.py
```

Or import it in your own code:

```python
from softmax_regression.model import SoftmaxRegression

model = SoftmaxRegression(learning_rate=0.1, n_epochs=1000, l2=0.0)
model.fit(X_train, y_train)
print(model.predict(X_test))        # class labels
print(model.predict_proba(X_test))  # probabilities, rows sum to 1
```

## How to run tests

From the repo root:

```bash
python -m pytest tests/test_softmax_regression.py -v
```
