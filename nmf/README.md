# NMF — Non-negative Matrix Factorization from Scratch

Factorizes a non-negative matrix `V` into two non-negative matrices
`W` and `H` such that `V ≈ W @ H`, using the Lee–Seung multiplicative
update rules. Numpy only, no other dependencies.

## The idea

NMF minimizes the squared Frobenius error `||V − W·H||²` with updates
that are pure multiplications by non-negative ratios:

```
H *= (Wᵀ V) / (Wᵀ W H + eps)
W *= (V Hᵀ) / (W H Hᵀ + eps)
```

Because each update multiplies by a non-negative factor, `W` and `H`
stay non-negative as long as they start non-negative — no projection
or clipping is needed. A small `eps` (1e-10) guards against division
by zero, and the factors start strictly positive (uniform in
[0.1, 1)) so no entry is stuck at zero forever.

## Why non-negativity gives interpretable parts

Unlike SVD/PCA, which allow negative entries and therefore
cancellations, NMF can only *add* parts together to rebuild the data.
Each component is therefore a meaningful, additive building block —
a topic in a document-term matrix, a facial feature in a face-image
matrix, a spectral signature in audio — rather than an abstract
direction with positive and negative lobes.

## Usage

```python
from nmf.model import NMF

model = NMF(n_components=3, max_iter=500, tol=1e-4, random_state=42)
W = model.fit_transform(V)   # W: (n_samples, 3)
H = model.components_        # H: (3, n_features)
model.reconstruction_err_    # final Frobenius error
model.n_iter_                # sweeps actually run

W_new = model.transform(V_new)        # encode new data, components_ fixed
recon = model.inverse_transform(W)    # W @ components_
model.score(V)  # explained variance (R²-like): 1 − ||V−WH||²/||V||²
```

`fit` validates its input and raises `ValueError` for 1D, empty,
negative, or non-finite input, and for `n_components` outside
`1 … min(V.shape)`.

## Run the demo

```bash
python nmf/demo.py
```

The demo builds a synthetic rank-3 matrix `V = W_true @ H_true`
(seed 42, shape 20×12, uniform entries) and factorizes it with
`n_components=3`. Output:

```
Original shape: (20, 12)
Relative reconstruction error: 0.000005
Iterations: 500
```

(Relative error = `||V − WH||_F / ||V||_F`; the run hits the 500
iteration cap with the error already at ~5e-6, far below the 0.05
assertion threshold.)

## Run the tests

```bash
python -m pytest tests/test_nmf.py -q
```

14 tests, all passing: rank-2 recovery with relative error < 0.05,
non-negativity of `W` and `H`, reconstruction error decreasing with
iterations, `fit_transform` / `transform` / `inverse_transform`
shapes and round-trip, `score` near 1 on exact data, and `ValueError`
for 1D, negative, empty, and NaN input plus `n_components=0` and
`n_components > min(shape)`.
