# PLS Classifier (PLS-DA)

Partial least squares discriminant analysis, from scratch with numpy only.

## Idea

PLS regression is fit with the NIPALS algorithm against one-hot class
targets, and a new sample is classified by the largest predicted target
value. PLS finds latent directions that maximise covariance between X and
the class targets, which makes it effective when features are collinear —
a case where ordinary classifiers can be unstable.

## Usage

```python
from pls_classifier.model import PLSClassifier

model = PLSClassifier(n_components=2).fit(X_train, y_train)
preds = model.predict(X_test)
print(model.explained_variance_ratio_)
```

## Parameters

- `n_components` (default 2): number of latent components, at most
  `min(n_samples - 1, n_features)`.
- `max_iter` (default 500), `tol` (default 1e-6): NIPALS stopping rules.
- `scale` (default True): standardise features to unit variance first.

## Demo

```bash
python pls_classifier/demo.py
```

Fits on 10 strongly collinear features built from 2 latent factors with
3 classes, and prints test accuracy for 1, 2 and 3 components.

## Tests

```bash
python -m pytest tests/test_pls_classifier.py -q
```
