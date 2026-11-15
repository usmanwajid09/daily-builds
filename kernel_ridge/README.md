# Kernel ridge regression, from scratch

Kernel ridge regression with an RBF (Gaussian) kernel — no
scikit-learn, just numpy. The prediction at a new point is a weighted
sum of kernel similarities to the training points,
`f(x) = sum_i alpha_i * k(x, x_i)`, and the dual weights have a
closed-form solution, `alpha = (K + lambda * I)^-1 y`, where `K` is
the kernel matrix on the training data and `lambda` is the ridge
regularisation strength. The RBF kernel,
`k(x, z) = exp(-gamma * ||x - z||^2)`, lets a linear method fit
genuinely non-linear relationships such as a sine wave. The kernel
matrix is solved with `np.linalg.solve` (falling back to a
least-squares solve if it is singular), and predictions need only the
stored training points and dual weights.

## Why

It is the simplest way to turn ridge regression non-linear: no
gradient descent, no iterations, just one matrix solve — and it is the
direct ancestor of support vector machines and Gaussian processes,
so understanding it pays off twice.

## Usage

```bash
pip install -r requirements.txt
python kernel_ridge/demo.py
```

This fits kernel ridge regression to a noisy sine wave (a non-linear
target a plain linear model cannot fit) on 80% of the data, and prints
test R^2 and MSE on the remaining 20%.

Example result:

```
Test R^2: 0.983
Test MSE: 0.0108
```

## Run tests

```bash
python -m pytest tests/test_kernel_ridge.py -v
```
