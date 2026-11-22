# One-Class SVM, from scratch

One-class SVM anomaly detection with an RBF kernel, built with numpy
only — no scikit-learn. Learns a boundary around the training data
using only the "normal" class: the score function is
`f(x) = sum_i alpha_i K(x_i, x)` and points with `f(x) < rho` are
flagged as anomalies. `alpha` and `rho` are fit by mini-batch SGD on
the one-class primal objective, with `alpha` projected back onto the
non-negative orthant after each step. After optimisation, `rho` is set
to the `nu`-quantile of the training scores, so roughly a fraction
`nu` of the training points fall outside the boundary.

## Usage

```bash
pip install -r requirements.txt
python one_class_svm/demo.py
```

This fits on a 200-point Gaussian blob and prints the inlier rate on
fresh normal points and the detection rate on far-away outliers.

## Run tests

```bash
python -m pytest tests/test_one_class_svm.py -v
```

## Demo output

```
Train size: 200, support vectors: 148
rho: 0.1604, gamma: 0.5527
Fresh inliers classified as normal: 0.88
Outliers detected as anomalies: 1.00
Mean decision score (inliers): 0.042
Mean decision score (outliers): -0.160
```
