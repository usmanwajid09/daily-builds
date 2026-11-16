# QDA Classifier — Quadratic Discriminant Analysis from scratch

Day 4 of the 60-day daily-builds streak. A Gaussian classifier where each
class gets its **own** covariance matrix, implemented with numpy only.

## What it is

`QDAClassifier` fits one multivariate Gaussian per class — a mean vector,
a covariance matrix, and a prior — then classifies a point by the highest
log-posterior discriminant:

```
score_k(x) = log prior_k - 0.5 * log|Sigma_k| - 0.5 * (x - mu_k)^T Sigma_k^-1 (x - mu_k)
```

The log-determinant uses `slogdet` and the Mahalanobis term uses `solve`
(no explicit inverse) for numerical stability. A ridge term `reg` is added
to each covariance diagonal, and a singular covariance falls back to a
pseudo-inverse. `predict_proba` is a softmax over the discriminants, with
the row max subtracted first so the rows sum to 1.

## Why it's useful

LDA assumes all classes share one covariance, so its boundaries are always
straight lines. QDA lets each class have its own spread, so its boundaries
are curved — it can separate classes that share a mean but differ in
variance (a tight cluster inside a wide one), which no linear classifier
can do. The trade-off: more parameters per class, so it wants more data.

## Usage

```python
from qda_classifier import QDAClassifier

clf = QDAClassifier(reg=1e-6)          # priors estimated from class frequencies
clf.fit(X_train, y_train)              # X: (n_samples, n_features), y: (n_samples,)
labels = clf.predict(X_test)           # class labels
probs = clf.predict_proba(X_test)      # rows sum to 1

print(clf.classes_)                    # sorted unique labels
print(clf.means_)                      # per-class means
print(clf.covariances_)                # per-class covariance matrices
print(clf.priors_)                     # per-class priors
```

Custom priors (must be positive and sum to 1):

```python
clf = QDAClassifier(priors=[0.2, 0.5, 0.3])
```

## Demo

```bash
python qda_classifier/demo.py
```

Generates 3 classes with different covariances (seed 42, 100/class),
trains on an 80/20 split, and prints train/test accuracy, per-class
covariance determinants, and the first 5 predictions vs true labels.
Test accuracy is >= 0.90.

## Tests

```bash
python -m pytest tests/test_qda_classifier.py -v
```

9 tests: accuracy >= 0.90 on 3 classes with different covariances;
`predict_proba` shape and row sums; QDA beating a shared-covariance
baseline on same-mean / different-variance data; a singular (duplicate
feature column) covariance handled without crashing; predict before fit
raises; invalid `reg`/priors and mismatched X/y raise `ValueError`;
string labels work.
