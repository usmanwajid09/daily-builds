# Linear SVM classifier, from scratch

A linear support-vector classifier — no scikit-learn, just numpy. It is
trained by stochastic gradient descent on the primal objective
`0.5 * ||w||^2 + C * sum(max(0, 1 - y_i (w.x_i + b)))`: the regularisation
term always shrinks the weights, and the hinge term pushes back only for
samples inside the margin. `decision_function` returns the signed score
whose sign is the prediction.

## Why

SVMs are the classic maximum-margin classifier: instead of just separating
the classes, they find the boundary with the widest gap between them.
Building one by hand shows how the hinge loss and the `C` parameter trade
margin width against training mistakes.

## Usage

```bash
pip install -r requirements.txt
python -m svm_from_scratch.demo
```

This generates two synthetic 2D clusters, fits an SVM with C=1.0 on 80%
of the data, and prints test accuracy on the remaining 20%.

## Run tests

```bash
python -m pytest tests/test_svm_from_scratch.py -v
```
