# LDA classifier, from scratch

Linear Discriminant Analysis — no scikit-learn, just numpy. LDA models
each class as a Gaussian with its own mean but a *shared* covariance
matrix, plus a class prior. A new point is scored by the log posterior
of each class,
`log prior_k - 0.5 * (x - mu_k)^T Sigma^-1 (x - mu_k)`,
and assigned to the highest-scoring class. Because the covariance is
shared, the quadratic terms cancel and the decision boundary between
any two classes is a hyperplane, which makes LDA fast, stable, and
surprisingly strong when classes are roughly Gaussian with similar
spread. The shared covariance is estimated from within-class scatter
and inverted with a pseudo-inverse, so it copes with a singular
covariance; an optional `shrinkage` parameter blends it toward a
scaled identity for extra stability on small samples.

## Why

LDA is the classic closed-form classifier: no iterations, no learning
rate to tune, and it gives calibrated class probabilities for free.
It is also a good baseline before reaching for heavier models — if LDA
already separates your classes well, you probably do not need a
neural net.

## Usage

```bash
pip install -r requirements.txt
python lda_classifier/demo.py
```

This generates three synthetic 2D Gaussian clusters, fits LDA on 80%
of the data, and prints test accuracy, class priors, and the first few
predicted probabilities on the remaining 20%.

Example result:

```
Test accuracy: 0.944
Class priors: [0.333, 0.333, 0.333]
```

## Run tests

```bash
python -m pytest tests/test_lda_classifier.py -v
```
