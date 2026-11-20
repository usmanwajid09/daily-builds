# RBF-Kernel SVM, from scratch

A binary support vector machine with a Gaussian (RBF) kernel, trained by
Platt's SMO (sequential minimal optimisation) on the dual problem — no
scikit-learn, just numpy. The RBF kernel lets the classifier separate
data that is not linearly separable, such as concentric circles or
two moons.

## How it works

The dual is maximised over multipliers `0 <= alpha <= C` with the
constraint `sum(alpha * y) == 0`. SMO updates two multipliers at a time
with an analytic step under the box constraints, and the bias is
recomputed as the average over the margin support vectors. `gamma`
controls the kernel width: `"scale"` (`1 / (n_features * X.var())`),
`"auto"` (`1 / n_features`), or a positive float.

## Usage

```python
from svm_rbf import RBFKernelSVM

model = RBFKernelSVM(C=1.0, gamma="scale", random_state=42)
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test, y_test))       # accuracy
print(model.n_support_)                  # number of support vectors
print(model.decision_function(X_test))   # signed distance to the hyperplane
```

## Run the demo

```bash
python svm_rbf/demo.py
```

This generates two concentric circles (inner radius ~1, outer radius ~3),
fits the SVM on 80% of the data, and prints train/test accuracy and the
number of support vectors.

## Run tests

```bash
python -m pytest tests/test_svm_rbf.py -v
```
