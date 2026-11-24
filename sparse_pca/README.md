# Sparse PCA

Principal components with L1-sparsity: each component is found by power
iteration on the covariance matrix, with a soft-thresholding step after
every multiplication so small loadings are driven to exactly zero.
Multiple components are extracted one at a time with deflation
(subtracting `lambda * v v^T` from the covariance), and every iterate is
projected onto the orthogonal complement of the components already found,
so the components stay mutually orthogonal and reconstruction error falls
monotonically as components are added.

Only numpy and the standard library are used.

## Model

- `SparsePCA(n_components=2, alpha=0.5, max_iter=100, tol=1e-6, random_state=None)`
  - `n_components`: number of components (>= 1, at most
    `min(n_samples, n_features)`).
  - `alpha`: sparsity strength (>= 0). `alpha=0` recovers (approximately)
    plain PCA; larger values zero out more loadings.
  - `max_iter` / `tol`: power-iteration budget and convergence tolerance.
  - `random_state`: seed for the random starting vectors.
- `fit(X)`: validates that `X` is 2D, non-empty and finite, centres it,
  then extracts the components. Raises `ValueError` for bad input.
- `transform(X)`: projects centred data onto the components.
- `inverse_transform(X_transformed)`: reconstructs data from scores.
- Attributes: `components_` (rows unit-norm), `explained_variance_`
  (`v^T Cov v` per component, non-increasing), `explained_variance_ratio_`,
  `mean_`, `sparsity_` (fraction of loadings that are exactly zero).

## Usage

```python
from sparse_pca.model import SparsePCA

pca = SparsePCA(n_components=2, alpha=0.5, random_state=0)
pca.fit(X)
print(pca.components_)      # sparse loadings, rows unit-norm
print(pca.sparsity_)        # fraction of exact-zero loadings
Z = pca.transform(X)
X_rec = pca.inverse_transform(Z)
```

## Demo

`demo.py` builds synthetic data (400 samples x 10 features) where only
the first 3 features share a strong latent factor and the other 7 are
pure noise, then fits with `alpha=0.5` and checks that the leading
component lands on the signal features, that some loadings are exactly
zero, and that reconstruction is good. Run it with:

```
python sparse_pca/demo.py
```

Output:

```
Data: 400 samples x 10 features (signal on features 0-2 only)

Leading component loadings:
  +0.566 +0.589 +0.576 -0.000 -0.000 -0.000 -0.000 -0.000 +0.000 -0.000
Fraction of zero loadings (all components): 0.80
Explained variance: [27.0701394   1.06189197]
Explained variance ratio: [0.75223879 0.02950839]
Reconstruction MSE (2 components): 0.7834

All demo checks passed.
```

The leading component puts ~equal weight on features 0-2 and exactly
zero on all 7 noise features, and explains 75% of the total variance.

## Tests

16 tests in `tests/test_sparse_pca.py` (run from the repo root):

```
python -m pytest tests/test_sparse_pca.py -q
```

They cover: `alpha=0` matching plain PCA's top eigenvector
(|cosine| > 0.99), sparsity increasing with `alpha`, unit-norm
components, transform shape, reconstruction error decreasing with more
components, explained variance positive and non-increasing, the sparse
leading component finding the signal features, determinism with the same
`random_state`, and `ValueError`s for 1D input, empty input, NaN input,
`n_components < 1`, `n_components > n_features`, `alpha < 0`, and
transform/inverse-transform before fitting.
