# K-Nearest Neighbours, from scratch

A tiny, dependency-light KNN classifier — no scikit-learn, just numpy.
Distance is Euclidean, the label is a majority vote of the k nearest
training points, and vote ties are broken by the nearest neighbour, so
predictions are deterministic.

## Why

KNN is one of the simplest classifiers to understand, and implementing it
by hand makes the trade-offs visible: choice of k, distance, and what
happens on ties and on unseen feature counts.

## Usage

```bash
pip install -r requirements.txt
python -m knn_from_scratch.demo
```

This generates two synthetic 2D clusters, fits KNN with k=5 on 80% of the
data, and prints test accuracy on the remaining 20%.

## Run tests

```bash
python -m pytest tests/test_knn.py -v
```
