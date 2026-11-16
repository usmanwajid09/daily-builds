# Texture Classifier (LBP + kNN)

Classify synthetic textures — stripes, checkerboard, noise, dots —
using Local Binary Pattern (LBP) histograms and a k-Nearest Neighbour
classifier. Each pixel is encoded by comparing its 8 neighbours to the
centre pixel; the normalised 256-bin histogram of those codes is the
texture's fingerprint.

## Why

LBP is a classic, fast texture descriptor that is invariant to monotonic
brightness changes, and it shows how a hand-crafted feature plus a simple
classifier can solve a vision task with no learned weights at all.

## Usage

```bash
pip install -r requirements.txt
python -m texture_classifier.demo
```

The demo generates 30 training and 15 test images per class, fits kNN
(k=3) on LBP histograms, and prints overall and per-class accuracy.

```python
from texture_classifier import TextureKNN, extract_features, make_texture

train = [make_texture("stripes"), make_texture("dots")]
clf = TextureKNN(k=3).fit(extract_features(train), ["stripes", "dots"])
print(clf.predict(extract_features([make_texture("dots")])))
```

## Run tests

```bash
python -m pytest tests/test_texture_classifier.py -v
```
