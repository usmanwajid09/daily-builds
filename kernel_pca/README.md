# Kernel PCA, from scratch

Kernel principal component analysis, built with numpy only — no
scikit-learn. The data is implicitly mapped to a high-dimensional feature
space via an RBF (or linear) kernel, the centred kernel matrix is
eigendecomposed, and the data is projected onto the leading eigenvectors.
The RBF kernel separates non-linear structure — e.g. concentric rings —
that plain PCA cannot.

## Usage

```bash
pip install -r requirements.txt
python kernel_pca/demo.py
```

This projects two synthetic concentric rings onto 2 kernel components and
prints the explained variance ratio and how well the first component
separates the rings.

## Run tests

```bash
python -m pytest tests/test_kernel_pca.py -v
```
