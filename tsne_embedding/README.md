# t-SNE Embedding — from scratch

t-distributed Stochastic Neighbour Embedding (t-SNE), implemented with
**numpy only** — no sklearn, no scipy.

## What it does

Reduces high-dimensional data to a low-dimensional embedding (usually 2D)
for visualisation. Points that are neighbours in the original space stay
neighbours in the embedding; well-separated clusters stay separated.

## How it works

1. **Input affinities** — for each point, a Gaussian kernel over its
   pairwise squared Euclidean distances gives conditional probabilities
   `p(j|i)`. A per-point bandwidth `sigma` is found by binary search so the
   Shannon entropy of that distribution equals `log(perplexity)`.
2. **Symmetrise** — `P = (P + P.T) / (2n)`, floored at `1e-12` and
   normalised to sum to 1.
3. **Embedding affinities** — Student-t kernel
   `q_ij = (1 + ||y_i - y_j||^2)^-1`, normalised, which has heavy tails so
   moderately distant points are not crushed together.
4. **Optimise** — gradient descent on the KL divergence `KL(P || Q)`,
   with momentum, early exaggeration (input affinities x12 for the first
   250 iterations), per-point gradient clipping to norm 4, and recentring
   after each step.

## Run

```bash
python tsne_embedding/demo.py
```

## Tests

```bash
pytest tests/test_tsne_embedding.py -v
```
