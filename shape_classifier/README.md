# Shape Classifier (Hu moments + kNN)

Classify synthetic shapes — circle, square, triangle, star — using the
first four Hu moment invariants, log-transformed and z-scored, with a
k-Nearest Neighbour classifier. Hu moments are (to first order)
invariant to translation, scale, and rotation, so the same shape is
recognised wherever it sits on the canvas and however big or rotated
it is. (The remaining three invariants vanish for mirror-symmetric
shapes, so their log transform is rasterisation noise and they are
left out of the classifier.)

## Why

Hu moments are a classic shape descriptor from the 1960s that still
shows up in OCR and object recognition. They demonstrate that a few
well-chosen invariants can replace a whole learned feature extractor
for simple shape tasks.

## Usage

```bash
pip install -r requirements.txt
python -m shape_classifier.demo
```

The demo generates 40 training and 20 test shapes per class with random
scale, position, and rotation, fits kNN (k=3) on log-Hu features, and
prints overall and per-class accuracy.

```python
from shape_classifier import ShapeKNN, extract_features, make_shape

train = [make_shape("circle"), make_shape("square")]
clf = ShapeKNN(k=3).fit(extract_features(train), ["circle", "square"])
print(clf.predict(extract_features([make_shape("star")])))
```

## Run tests

```bash
python -m pytest tests/test_shape_classifier.py -v
```
