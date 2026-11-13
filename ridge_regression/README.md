# Ridge Regression, from scratch

Ridge regression (L2-regularised linear regression) solved in closed
form with numpy only — no scikit-learn. The intercept is fitted by
centring the data and is not penalised; larger `alpha` shrinks the
coefficients towards zero, which helps when features are correlated or
the sample is small. With `alpha=0` it reduces to ordinary least squares.

## Usage

```bash
pip install -r requirements.txt
python -m ridge_regression.demo
```

This generates synthetic noisy data from `y = 3*x0 - 2*x1 + 5`, fits the
model with `alpha` 0, 1 and 10, and prints the test R^2 and coefficients
for each.

## Run tests

```bash
python -m pytest tests/test_ridge_regression.py -v
```
