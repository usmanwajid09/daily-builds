# PCA, from scratch

A tiny principal component analysis implementation — no scikit-learn, just
numpy. The data are centred, the covariance matrix is formed, and its
eigenvectors (sorted by decreasing eigenvalue) become the principal
components. `transform` projects onto the first `n_components` components
and `inverse_transform` reconstructs the data from them, so you can see
exactly how much information a low-dimensional projection keeps.

## Why

PCA is the standard way to compress and visualise high-dimensional data.
Building it by hand shows that it is just an eigendecomposition of the
covariance matrix, and that the explained variance ratio tells you how
many components are worth keeping.

## Usage

```bash
pip install -r requirements.txt
python -m pca_from_scratch.demo
```

This generates correlated synthetic 4D data, keeps the first 2 principal
components, and prints the variance they explain plus the reconstruction
error.

## Run tests

```bash
python -m pytest tests/test_pca_from_scratch.py -v
```
