# Leaf Disease Detector (synthetic leaves, kNN)

Classify synthetic leaf images as **healthy**, **spot_disease** (brown
spots), or **yellowing** (chlorosis) using colour + texture features and
a k-nearest-neighbours classifier (k=3, Euclidean distance on
standardised features). All images are generated in code with numpy
only — no PIL/OpenCV, no external files.

## How it works

1. **Generation** (`make_leaf`): a green ellipse leaf on a neutral
   background, with a centre vein, a slight gradient and noise.
   Diseased variants add brown circular spots, or yellow patches where
   the green shifts toward yellow.
2. **Features** (`extract_features`): a 13-value vector — mean/std of
   R, G, B; mean normalised channel ratios; fractions of brown, yellow
   and green pixels; and a brown-blob count (4-connected components
   with ≥3 pixels).
3. **Classification** (`LeafDiseaseClassifier`): kNN over the
   standardised feature vectors. Majority vote, ties broken by the
   closest neighbour's class.

## Usage

```bash
python leaf_disease_detector/demo.py
```

```python
from leaf_disease_detector import LeafDiseaseClassifier, make_dataset, make_leaf

train_images, train_labels = make_dataset(30, seed=7)
clf = LeafDiseaseClassifier(k=3).fit(train_images, train_labels)
clf.predict(make_leaf("spot_disease", rng=0))  # "spot_disease"
```

## Demo output summary

Trained on 90 leaves (30 per class, seed 7) and tested on 45 held-out
leaves (15 per class, seed 99):

- overall accuracy: **1.000 (45/45 correct)**
- healthy: 1.00, spot_disease: 1.00, yellowing: 1.00
- example predictions: all three classes predicted correctly

## Run tests

```bash
python -m pytest tests/test_leaf_disease_detector.py -q
```

21 tests: leaf shapes/values, feature length and determinism, class
separation by brown/yellow fractions, classifier fit/predict, the
≥0.90 held-out accuracy threshold, and ValueError paths for bad
labels, sizes, images, k, and empty/mismatched training sets.
