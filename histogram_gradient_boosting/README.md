# Histogram Gradient Boosting Regressor, from scratch

Gradient boosting for squared loss, built with numpy only — no
scikit-learn. Each feature is binned into at most `max_bins`
equal-frequency bins before training, and split finding works on bin
histograms (sums of gradients and Hessians per bin) instead of sorting
every sample at every node — the core idea behind LightGBM-style
histogram boosting. For squared loss the gradient is `pred - y` and
the Hessian is 1, so a leaf's optimal value is
`-sum(grad) / (sum(hess) + lambda)` and splits use the second-order
gain with `gamma` (`min_split_gain`) and `lambda` penalties.

## Usage

```bash
pip install -r requirements.txt
python histogram_gradient_boosting/demo.py
```

This fits 200 shallow trees on noisy sinusoidal data and prints the
test R^2 plus the training MSE after the first and last stages.

## Run tests

```bash
python -m pytest tests/test_histogram_gradient_boosting.py -v
```

## Demo output

```
Train size: 320, test size: 80
Histogram GB test R^2: 0.982
Train MSE after 1 tree: 1.0366
Train MSE after 200 trees: 0.0090
```
