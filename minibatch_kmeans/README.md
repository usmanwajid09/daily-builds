# MiniBatch K-Means

Mini-batch K-Means clustering built from scratch with numpy only.

## What it does

Like K-Means, but each step updates the centres from a small random
mini-batch instead of the whole dataset, so it scales to data that
does not fit comfortably in memory and converges in far fewer full
passes. Each centre keeps a running count of the points assigned to
it; the learning rate for a centre is 1 / count, so early updates move
fast and later ones settle. Centres start from k-means++ (or random)
initialisation. `partial_fit` supports online learning, one batch at
a time.

## How to run

```bash
python minibatch_kmeans/demo.py
```

```python
from minibatch_kmeans.model import MiniBatchKMeans

model = MiniBatchKMeans(n_clusters=3, batch_size=50, max_iter=100,
                        random_state=42)
model.fit(X)
print(model.labels_)
print(model.inertia_)
print(model.predict(X_new))
```

## Demo result

Three synthetic 2D clusters (100 points each, centres (0,0), (5,0),
(2.5,4.5)), batch size 50, 100 iterations.

- Inertia: **359.83**
- Cluster purity: **0.997**
- Found centres within ~0.1 of the true centres.

## Example output

```
Inertia:        359.83
Iterations run: 90
Cluster purity: 0.997
Found centres:
  [2.39, 4.44]
  [5.05, 0.03]
  [-0.0, -0.07]
True centres:
  [0.0, 0.0]
  [5.0, 0.0]
  [2.5, 4.5]
Cluster sizes: [101, 100, 99]
```

## Tests

11 pytest tests in `tests/test_minibatch_kmeans.py` — cluster
recovery and purity, inertia decreasing from initialisation, predict
matching labels, new points near true centres landing in distinct
clusters, transform distances, `partial_fit` updates and
initialisation, random init, and input validation.
