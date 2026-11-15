# Gaussian process regression, from scratch

Gaussian process (GP) regression — no scikit-learn, just numpy. A GP
places a prior over functions: any finite set of function values is
jointly Gaussian with covariance given by a kernel. Conditioning that
prior on noisy observations gives a closed-form posterior at new
points, with both a predictive mean and a predictive variance — so the
model knows how uncertain it is, growing less sure far from the
training data. This module uses the RBF (squared exponential) kernel,
`k(x, z) = sigma_f^2 * exp(-||x - z||^2 / (2 * length_scale^2))`, plus
observation noise `sigma_n^2`. The training covariance is factorised
once with a Cholesky decomposition (numerically safer than inverting
the matrix), and predictions reuse that factorisation. `predict`
returns both the mean and the standard deviation;
`log_marginal_likelihood` scores the fit of the chosen
hyperparameters, which is what makes kernel tuning principled.

## Why

Most regressors give a point prediction and no idea how much to trust
it. A GP gives a full predictive distribution, which is why it is the
go-to model for Bayesian optimisation, active learning, and any
small-data problem where knowing what you don't know matters.

## Usage

```bash
pip install -r requirements.txt
python gaussian_process_regressor/demo.py
```

This fits a GP to a noisy sine wave on 80% of the data, and prints
test R^2 and MSE, the mean predictive uncertainty on the test points,
and the (larger) uncertainty far outside the training range, on the
remaining 20%.

Example result:

```
Test R^2: 0.982
Test MSE: 0.0115
Mean predictive std on test points: 0.123
Predictive std far from data (x=±10): [1.0, 1.0]
Log marginal likelihood: -5.63
```

## Run tests

```bash
python -m pytest tests/test_gaussian_process_regressor.py -v
```
