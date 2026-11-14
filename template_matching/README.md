# Template Matching, from scratch

Find where a small template image occurs inside a larger image, using
normalised cross-correlation (NCC) — no OpenCV, just numpy. At every
candidate position the image patch is compared with the template
after removing each one's mean and dividing by each one's standard
deviation, so the score is unaffected by a uniform brightness shift or
contrast rescale of the patch.

## Why

Template matching is the simplest form of object detection: given a
reference patch (a logo, a digit, a landmark), scan the image for the
spot that looks most like it. NCC is the classic score because raw
squared error is fooled by brightness — a bright patch can beat the
true match simply by being bright — while NCC measures shape, not
level. It also shows why correlation is computed from centred,
normalised patches, and where it breaks down: a flat (constant)
template has no shape to correlate, so it scores 0 everywhere.

## Usage

```bash
pip install -r requirements.txt
python -m template_matching.demo
```

This builds a synthetic 60x80 random-noise image, pastes a 12x10
patch into it at a known location, and prints the found location
against the true one plus the NCC score (1.0 for an exact copy).

Use it on your own images from Python:

```python
import numpy as np
from template_matching import match_template, match_template_map

image = np.load("photo_gray.npy")      # 2D grayscale array
template = np.load("patch_gray.npy")   # 2D grayscale array, smaller
row, col, score = match_template(image, template)
print(f"best match at top-left ({row}, {col}), score {score:.3f}")

scores = match_template_map(image, template)  # full NCC map
```

`match_template` returns the top-left corner of the best-matching
patch and its score in [-1, 1] (1 = exact copy up to an affine
brightness change, -1 = exact inverted copy, 0 = uncorrelated).
`match_template_map` returns the whole score map of shape
`(H - th + 1, W - tw + 1)`. `ncc_at(image, template, row, col)`
scores a single position.

## Edge cases

- **Constant template** (zero variance, e.g. an all-black patch):
  correlation is undefined, so it scores 0 everywhere — never NaN.
  A constant image patch scores 0 the same way.
- **Template larger than the image**: `ValueError`.
- Inputs must be 2D grayscale arrays with a numeric dtype, non-empty,
  and finite; anything else raises `ValueError` with a helpful message.

## Run tests

```bash
python -m pytest tests/test_template_matching.py -v
```
