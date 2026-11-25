# FastICA

Blind source separation from scratch — numpy only, no sklearn/scipy.

## What it does

Given a linear mixture `X = S @ A.T` of independent non-Gaussian sources,
FastICA recovers the sources (up to permutation and sign) and the mixing
matrix, using only the observed mixtures.

## How it works

1. **Centre** `X` (subtract the per-feature mean).
2. **Whiten** via the eigendecomposition of the covariance matrix, so the
   whitened data has identity covariance (tiny eigenvalues are clipped).
3. **Fixed-point iteration** with the tanh nonlinearity
   (`g(u) = tanh(u)`, `g'(u) = 1 - tanh(u)^2`), with symmetric
   decorrelation `W <- (W W^T)^(-1/2) W` after every step, until the
   unmixing matrix stops changing.

Gaussian sources cannot be separated this way — the algorithm looks for
directions of maximal non-Gaussianity, and a Gaussian has none.

## Run

```bash
# from anywhere (the demo bootstraps its own path)
python fastica/demo.py

# tests (from the repo root)
pytest tests/test_fastica.py -v
```

## Demo result

Mean best-match |correlation| between the true sources (uniform +
Laplace, mixed with a fixed 2x2 matrix) and the recovered components is
≈ 1.0, with a near-zero reconstruction error on the round trip through
`inverse_transform`.

## API

```python
from fastica import FastICA

ica = FastICA(n_components=2, random_state=42)
S = ica.fit_transform(X)      # independent components
X_back = ica.inverse_transform(S)
ica.components_               # unmixing matrix (whitened space)
ica.mixing_                   # mixing matrix
ica.mean_                     # per-feature mean removed in fit
```
