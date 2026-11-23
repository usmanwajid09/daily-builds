# Traffic Sign Classifier (synthetic signs + kNN)

Classify synthetic traffic signs by colour + shape features with a
k-nearest-neighbours classifier (k=3, Euclidean distance on
standardised features). All images are generated programmatically as
small RGB images (numpy only, no PIL/OpenCV, no external files).

## Classes

| Class | Appearance |
|---|---|
| `stop` | Red octagon, white border ring, white centre bar |
| `yield` | Red-bordered inverted triangle, white inside |
| `speed_limit` | White circle, red border ring, dark centre bar |
| `no_entry` | Red circle, white horizontal bar |
| `go` | Blue circle, white up-arrow |
| `warning` | Yellow upright triangle, black exclamation mark |

Each variant adds scale/translation jitter, background variation and
a little Gaussian noise, so the classifier must generalise rather than
memorise one exact picture.

## How it works

1. `make_sign(sign_type, rng, size=32)` draws a sign with numpy masks
   (circle via distance mask, triangle via barycentric test, octagon
   via a clipped-square mask).
2. `extract_features(image)` builds a 62-length vector: 6 colour
   proportions (red/blue/yellow/white/black/grey), 8 shape statistics
   of the foreground mask (fill ratio, aspect ratio, extent, corner
   occupancy, left-right / top-bottom symmetry, centroid offset) and
   a coarse 4x4 downsampled colour histogram (48 values, which also
   captures spatial layout — e.g. yield points down, warning up).
3. `TrafficSignClassifier(k=3)` standardises features with the
   training mean/std and classifies by majority vote of the k nearest
   training examples.

## Usage

```bash
python traffic_sign_classifier/demo.py
```

```python
from traffic_sign_classifier import (
    TrafficSignClassifier, accuracy, make_dataset, make_sign,
)

train_images, train_labels = make_dataset(60, seed=1)
test_images, test_labels = make_dataset(30, seed=2)

clf = TrafficSignClassifier(k=3)
clf.fit(train_images, train_labels)
print(accuracy(test_images, test_labels, clf))  # 1.0
print(clf.predict_one(make_sign("stop", rng=7)))  # 'stop'
```

## Demo output

```
train: 360 images (60 per class), test: 180 images (30 per class)
overall accuracy: 1.000 (180/180 correct)
  stop        : accuracy 1.00
  yield       : accuracy 1.00
  speed_limit : accuracy 1.00
  no_entry    : accuracy 1.00
  go          : accuracy 1.00
  warning     : accuracy 1.00
example predictions:
  true=stop         predicted=stop
  true=warning      predicted=warning
  true=go           predicted=go
```

## Run tests

```bash
python -m pytest tests/test_traffic_sign_classifier.py -q
# 21 passed
```
