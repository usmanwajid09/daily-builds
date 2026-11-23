# Ridge Classifier

Ridge regression adapted as a classifier, from scratch with numpy only.

## Idea

Fit ridge regression (closed-form least squares with an L2 penalty) against
one-hot class targets coded as -1/+1, then classify by the largest decision
value. Two classes: the sign of the decision function. More classes:
one-vs-rest with argmax.

## Usage

```python
from ridge_classifier.model import RidgeClassifier

model = RidgeClassifier(alpha=1.0).fit(X_train, y_train)
preds = model.predict(X_test)
scores = model.decision_function(X_test)
print(model.score(X_test, y_test))
```

## Parameters

- `alpha` (default 1.0): L2 regularisation strength, must be positive.
  The intercept is never penalised.
- `fit_intercept` (default True): whether to fit an intercept term.

## Demo

```bash
python ridge_classifier/demo.py
```

Trains on three synthetic 2D clusters and prints test accuracy for
alpha = 0.1, 1.0 and 10.0.

## Tests

```bash
python -m pytest tests/test_ridge_classifier.py -q
```
