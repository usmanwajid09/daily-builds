# Digit Classifier — HOG + kNN

Classify synthetic 8x8 digit images (0–9) with Histogram of Oriented
Gradients (HOG) features and a k-nearest-neighbours classifier
(k=3, Euclidean distance), and compare it against raw-pixel
nearest-template matching on the same noisy images.

## How it works

1. **Render** — each digit is rendered from the hand-defined 8x8
   binary templates in `digit_recognizer/digits.py`, then corrupted
   with random pixel flips (`flip_prob=0.05` by default).
2. **HOG features** — gradients are computed with central
   differences; orientations (unsigned, 0–180°) are binned into 9
   bins, each pixel voting with its gradient magnitude. Cells of
   2×2 pixels form orientation histograms; overlapping 2×2-cell
   blocks are L2-normalised and concatenated into a 324-dim vector
   (`cell_size=4` gives a 36-dim vector instead).
3. **kNN** — a new image is classified by majority vote of its 3
   nearest training vectors; ties go to the smallest digit, so
   predictions are deterministic.
4. **Baseline** — `template_match_accuracy` scores the same images
   by nearest-template matching on raw pixels, so the two approaches
   can be compared head to head.

## Usage

```bash
python digit_classifier_hog/demo.py
```

```python
from digit_classifier_hog import hog_features, make_dataset, train, accuracy

train_f, train_y = make_dataset(30, flip_prob=0.05, seed=1, cell_size=2)
test_f, test_y = make_dataset(20, flip_prob=0.05, seed=99, cell_size=2)
clf = train(train_f, train_y, k=3)
print(accuracy(test_f, test_y, clf))

# One image:
from digit_recognizer.digits import template
print(clf.predict_one(hog_features(template(7), cell_size=2)))  # 7
```

`make_dataset(..., append_pixels=True)` appends the flattened raw
image to the HOG vector, which closes most of the gap to template
matching (see below).

## Demo output

```
train: 300 noisy variants (30 per digit, flip_prob=0.05)
test:  200 held-out noisy variants (20 per digit)
HOG descriptor: cell_size=2, length 324
HOG + kNN (k=3):        0.960 (192/200 correct)
HOG + pixels + kNN:     0.970 (194/200 correct)
template matching:      0.985 (197/200 correct)
```

Honest verdict: template matching wins on these synthetic digits.
The noisy images are pixel flips of the exact templates, so
raw-pixel distance to the clean prototype is near-optimal, while
HOG's edge histograms discard the absolute-position detail that
makes templates so strong. HOG + kNN is still a real, general
classifier — unlike template matching it learns from examples and
would keep working on digits drawn in a style with no clean
template available.

## Run tests

```bash
python -m pytest tests/test_digit_classifier_hog.py -q
```
