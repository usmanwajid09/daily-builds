# Truncated SVD

Truncated singular value decomposition built from scratch with numpy
only, via eigendecomposition of the Gram matrix.

## What it does

Keeps only the top `n_components` singular values and vectors — the
standard low-rank approximation / dimensionality reduction for data
that should not be centred (unlike PCA, e.g. sparse term-document
matrices). Exposes `components_`, `singular_values_`,
`explained_variance_`, `explained_variance_ratio_`, plus `transform`,
`inverse_transform`, and `reconstruction_error`.

## How to run

```bash
python truncated_svd/demo.py
```

## Demo result

Synthetic rank-3 data (200 samples, 10 features) plus small noise,
keeping 3 components.

- Total variance explained by 3 components: **0.997**
- Reconstruction MSE with 3 components: **0.0069**
- Reconstruction MSE with 1 component: 1.3928
