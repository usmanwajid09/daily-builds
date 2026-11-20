# neural_net_regressor

A multi-layer perceptron (MLP) regressor built from scratch with numpy
only: one hidden layer with ReLU activation, a linear output, mini-batch
gradient descent with backpropagation on MSE, Adam (default) or SGD, and
early stopping on a held-out validation split that restores the best
weights. Features and the target are standardized internally.

## Usage

```python
import numpy as np
from neural_net_regressor.model import MLPRegressor

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(200, 1))
y = np.sin(X.ravel()) + 0.1 * rng.normal(size=200)

model = MLPRegressor(hidden_units=32, learning_rate=0.01, epochs=500,
                     batch_size=32, random_state=42)
model.fit(X, y)
print(model.score(X, y))          # R^2
print(model.predict(X[:5]))       # 1D array of predictions
print(model.n_epochs_)            # epochs actually run (early stopping)
print(model.history_)             # train MSE per epoch
```

Key parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `hidden_units` | 32 | units in the single hidden layer |
| `learning_rate` | 0.01 | optimizer step size |
| `epochs` | 500 | maximum epochs |
| `batch_size` | 32 | mini-batch size |
| `optimizer` | `"adam"` | `"adam"` or `"sgd"` |
| `patience` | 20 | epochs without val-loss improvement before stopping; `0` disables early stopping |
| `random_state` | `None` | seed for reproducible initialization/shuffling |
| `l2` | 0.0 | weight-decay penalty |
| `validation_fraction` | 0.2 | fraction held out for early stopping |

## Run the demo

From the repo root:

```bash
python neural_net_regressor/demo.py
```

Fits a noisy sine curve (200 points, seed 42) and prints train/test R^2
plus a few predictions against the true values.

## Run the tests

```bash
pytest tests/test_neural_net_regressor.py -q
```
