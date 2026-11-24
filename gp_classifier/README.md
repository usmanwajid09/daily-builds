# gp_classifier

Gaussian process classifier with a Laplace approximation and an RBF
(squared-exponential) kernel. NumPy + standard library only — no new
dependencies.

## Model

- **Kernel:** RBF, `k(x, x') = signal_variance * exp(-||x - x'||² / (2 * length_scale²))`.
- **Likelihood:** Bernoulli with a sigmoid link on latent function values `f`.
- **Inference:** Laplace approximation. Newton iterations on the latent
  `f` (Rasmussen & Williams, Algorithm 3.1): `W = sigmoid(f)(1 - sigmoid(f))`,
  each step solves `(K⁻¹ + W)` via the Cholesky factor of
  `B = I + W^{1/2} K W^{1/2}`, with jitter added for stability.
  Converges when the largest change in `f` drops below `tol`.
- **Prediction:** latent mean `K_*ᵀ α`, latent variance
  `K_** − k_*ᵀ (K + W⁻¹)⁻¹ k_*` (computed from the same Cholesky factor
  of `B`). Probabilities use the MacKay probit approximation,
  `π ≈ sigmoid(mean / sqrt(1 + π/8 · var))`, so points far from the
  training data revert towards 0.5 as the latent variance grows.

`GaussianProcessClassifier(length_scale=1.0, signal_variance=1.0,
max_iter=50, tol=1e-5)` — both kernel parameters must be > 0.

## Usage

```python
from gp_classifier.model import GaussianProcessClassifier

model = GaussianProcessClassifier(length_scale=1.0, signal_variance=1.0)
model.fit(X_train, y_train)          # y in {0, 1}
labels = model.predict(X_test)       # 0/1 array
proba = model.predict_proba(X_test)  # shape (n, 2), rows sum to 1
accuracy = model.score(X_test, y_test)
```

`fit` raises `ValueError` for non-2D / non-finite `X`, non-binary labels,
mismatched lengths, or fewer than 2 samples. `predict`, `predict_proba`
and `score` raise `ValueError` before fitting.

## Demo

Run `python gp_classifier/demo.py` (from the repo root). It fits on two
well-separated Gaussian blobs (seed 42, centres ±3) and prints:

```
Train size: 160, test size: 40
Newton iterations: 6
Train accuracy: 1.000
Test accuracy: 1.000
Mean P(y=1) on true class 0: 0.113
Mean P(y=1) on true class 1: 0.860
P(y=1 | class 0 centre [-3, 0]): 0.051
P(y=1 | class 1 centre [3, 0]): 0.938
P(y=1 | far from all data [0, 10]): 0.500
All demo asserts passed.
```

## Tests

15 tests in `tests/test_gp_classifier.py` (run from the repo root):

```
python -m pytest tests/test_gp_classifier.py -q
```

They cover accuracy > 0.9 on separable blobs, probabilities in [0, 1]
with rows summing to 1, probabilities near 0/1 deep inside each class,
predict shape/dtype, score in [0, 1], `ValueError`s for 1D `X`,
mismatched lengths, empty input, NaN, non-binary labels, a single
sample, `length_scale`/`signal_variance` ≤ 0, predicting before fit,
determinism (same fit twice → same predictions), and predict/proba
agreement at the 0.5 threshold.
