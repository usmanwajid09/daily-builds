# Kernel Logistic Regression

Kernel logistic regression with an RBF kernel, from scratch with numpy only.

## Idea

Logistic regression in the dual: the decision function is a weighted sum
of RBF kernel evaluations against the training points,
`f(x) = sum_i alpha_i * k(x_i, x) + b`, with the coefficients trained by
gradient descent on the L2-regularised logistic loss. The kernel trick
lets it learn non-linear decision boundaries without ever building the
feature map explicitly. Binary only.

## Usage

```python
from kernel_logistic_regression.model import KernelLogisticRegression

model = KernelLogisticRegression(gamma=2.0, C=1.0).fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

## Parameters

- `gamma` (default 1.0): RBF kernel width, `k(x, z) = exp(-gamma * ||x-z||^2)`.
- `C` (default 1.0): inverse regularisation strength.
- `max_iter` (default 1000), `tol` (default 1e-4): stopping rules.
- `learning_rate` (default None): gradient step size; None computes the
  largest safe step (1 / Lipschitz constant) automatically.

## Demo

```bash
python kernel_logistic_regression/demo.py
```

Fits on the two-moons dataset (not linearly separable) and prints test
accuracy for gamma = 0.5, 2.0 and 8.0.

## Tests

```bash
python -m pytest tests/test_kernel_logistic_regression.py -q
```
