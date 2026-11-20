# Gesture Classifier (Hu moments + kNN)

Classify synthetic hand gestures — circle, square, triangle, star —
drawn as filled binary masks with random scale, position and rotation.
Each gesture is described by three rotation/scale/translation
invariant features — the log-transformed first Hu moment invariant,
the radial spread of boundary pixels, and the solidity (area /
convex-hull area) — and a k-Nearest Neighbour classifier separates
the classes. `predict_proba` returns the kNN vote share per class, so
the confidence of each prediction is visible.

## Usage

```python
from gesture_classifier import (
    GestureClassifier, accuracy, extract_features, make_gesture,
)

train = [make_gesture("circle"), make_gesture("square")]
clf = GestureClassifier(k=3).fit(extract_features(train),
                                 ["circle", "square"])
print(clf.classify(make_gesture("star")))
print(clf.predict_proba(extract_features([make_gesture("circle")])))
```

## Run demo / tests

```bash
python gesture_classifier/demo.py
python -m pytest tests/test_gesture_classifier.py -v
```

The demo generates 30 training and 15 test gestures per class, fits
kNN (k=3) on the three invariant features, and prints overall and
per-class accuracy.
