# Label Propagation

Semi-supervised label propagation on a k-nearest-neighbour graph, numpy only.

## What it does

Given a dataset where only a few points are labelled (`y = -1` means
unlabeled), it spreads those labels across the graph: points that are close
to each other end up with similar labels. With just 10 labels per class on
two well-separated blobs it recovers the true labels of the remaining
points with > 0.9 accuracy.

## How it works

1. Build a kNN graph with Gaussian (RBF) weights
   `w_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))`, where sigma is the mean
   distance to the k-th neighbour. Symmetrise and row-normalise to a
   transition matrix `S = D^-1 W`.
2. Start from `F = Y`, the one-hot matrix of the labelled points (zeros for
   unlabeled points).
3. Iterate `F <- alpha * S @ F + (1 - alpha) * Y`, hard-clamping the labelled
   rows back to `Y` each iteration, until the max change drops below `tol`.
4. Predicted label = `argmax` of each row of `F`
   (`label_distributions_`, rows sum to 1).

`predict(X)` is transductive: for the fitted training points it returns the
propagated labels; for new points it takes a Gaussian-weighted vote of the
k nearest training points' label distributions.

## Run

```bash
python label_propagation/demo.py
pytest tests/test_label_propagation.py -v
```
