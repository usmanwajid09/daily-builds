# Elastic Net Classifier

Logistic regression with an elastic-net (L1 + L2) penalty, from scratch
with numpy only.

## Idea

Trained with proximal gradient descent (ISTA): each step takes a gradient
step on the smooth part (logistic loss + L2 penalty) and then applies the
soft-thresholding operator for the L1 penalty, which can drive coefficients
to exactly zero and so select features. Multiclass problems are handled
one-vs-rest. Features are standardised internally so the penalty treats
every feature on the same scale.

## Usage

```python
from elastic_net_classifier.model import ElasticNetClassifier

model = ElasticNetClassifier(alpha=0.5, l1_ratio=0.5).fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

## Parameters

- `alpha` (default 1.0): overall penalty strength (>= 0).
- `l1_ratio` (default 0.5): 1.0 is pure L1 (lasso), 0.0 is pure L2 (ridge).
- `max_iter` (default 1000), `tol` (default 1e-4): stopping rules.
- `learning_rate` (default None): proximal-gradient step size; None
  computes the largest safe step (1 / Lipschitz constant) automatically.

## Demo

```bash
python elastic_net_classifier/demo.py
```

Fits on two synthetic clusters with two informative features and four
noise features, and shows accuracy plus how many coefficients the L1
penalty zeroes out for `l1_ratio` 0.0, 0.5 and 1.0.

## Tests

```bash
python -m pytest tests/test_elastic_net_classifier.py -q
```
