# Factor Analysis (EM)

Factor analysis via the expectation–maximisation algorithm, implemented
from scratch with numpy + stdlib only.

## Model

```
x = W @ z + mu + eps,   z ~ N(0, I),   eps ~ N(0, Psi),   Psi diagonal
```

- `W` (shape `n_features x n_factors`) is the **loading matrix**:
  how strongly each latent factor drives each observed feature.
- `z` is the low-dimensional latent factor score for each sample.
- `Psi` is per-feature **noise variance** (unique variance not shared
  with other features); `mu` is the feature mean.

Unlike PCA, factor analysis models noise explicitly, so it separates
shared (common-factor) variance from per-feature noise.

## EM algorithm

- **E-step:** posterior `z | x ~ N(m, G)` with
  `G = (I + W^T Psi^-1 W)^-1` and `m = G W^T Psi^-1 (x - mu)`.
- **M-step:** closed-form updates for `W` and `Psi` from the expected
  sufficient statistics; `Psi` is clipped to a floor of `1e-6` so it
  stays positive. `mu` is the sample mean.
- **Convergence:** stops when the log-likelihood improves by less than
  `tol` (default `1e-4`), or after `max_iter` iterations.

Loadings are only identified up to rotation and sign — the implied
covariance `W W^T + Psi` is what the model pins down.

## Usage

```python
from factor_analysis.model import FactorAnalysis

model = FactorAnalysis(n_factors=2, max_iter=200, random_state=0).fit(X)
scores = model.transform(X)          # posterior mean of z, shape (n, 2)
recon = model.reconstruct(X)         # W @ scores + mu
print(model.components_)             # loadings W
print(model.noise_variance_)         # Psi diagonal
print(model.log_likelihood(X))       # total log-likelihood
print(model.score(X))                # mean per-sample log-likelihood
```

## Demo

```
$ python factor_analysis/demo.py
```

Generates 300 samples of 6 features from a known 2-factor structure
(fixed seed), fits the model and prints the recovered loadings next to
the true ones, the noise variances, reconstruction MSE and final
log-likelihood. Example output:

```
Samples: 300, features: 6, factors: 2, EM iterations: 19
Recovered noise variance: [0.232, 0.288, 0.28, 0.209, 0.25, 0.255]
Factor-score shape: (300, 2)
Reconstruction MSE: 0.1803
Covariance MSE vs truth: 0.0064
Final log-likelihood: -1951.19
All demo checks passed.
```

## Tests

```
$ python -m pytest tests/test_factor_analysis.py -q
```

14 tests cover loadings/covariance recovery on synthetic data,
reconstruction MSE, transform shape, log-likelihood increasing across
EM iterations, positive noise variance, determinism with the same
`random_state`, and `ValueError`s for 1D input, empty input, NaN/inf,
mismatched feature counts, `n_factors >= n_features`, `n_factors < 1`,
too few samples, and transforming before fitting.
