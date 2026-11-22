# Bayesian Logistic Regression, from scratch

Bayesian logistic regression with a Laplace approximation, built with
numpy only — no scikit-learn. The model is `p(y = 1 | x) =
sigmoid(w^T x)` with a Gaussian prior `w ~ N(0, alpha^-1 I)`. The
posterior is approximated by a Gaussian centred at the MAP estimate
(found by Newton-Raphson / IRLS) with covariance equal to the inverse
Hessian of the negative log posterior at the MAP. Predictive
probabilities use MacKay's probit approximation, which folds the
posterior uncertainty into the prediction — points far from the
training data get probabilities pulled towards 0.5, so the model
knows what it does not know.

## Usage

```bash
pip install -r requirements.txt
python bayesian_logistic_regression/demo.py
```

This fits on two synthetic Gaussian blobs and prints the test
accuracy, the MAP weights, and predictive probabilities for a point
near the data and one far away.

## Run tests

```bash
python -m pytest tests/test_bayesian_logistic_regression.py -v
```

## Demo output

```
Train size: 160, test size: 40
Test accuracy: 0.925
Newton iterations: 9
MAP coefficients: [2.79, -0.215], intercept: -0.28
P(y=1 | near point [0, 0]): 0.433 (MAP plug-in: 0.430)
P(y=1 | far point [0, 50]): 0.303 (MAP plug-in: 0.000)
```
