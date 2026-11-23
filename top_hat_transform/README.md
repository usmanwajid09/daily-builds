# Top-Hat Transform, from scratch

Morphological top-hat transforms with no OpenCV, just numpy. The
white top-hat is `image - opening(image)` and extracts bright details
smaller than the structuring element; the black top-hat is
`closing(image) - image` and extracts dark details smaller than the
structuring element. Square and disk structuring elements are
supported, built on grayscale erosion and dilation.

## Why

Top-hats are the classic way to pull small features out of an uneven
background: opening removes bright spots smaller than the element
while leaving the background alone, so subtracting it leaves only the
spots. It is the same idea behind illumination correction and
small-defect detection — a bright scratch on a shaded surface, or a
dark speck on a bright one, pops out while the slow background
gradient cancels to zero.

## Usage

```bash
pip install -r requirements.txt
python top_hat_transform/demo.py
```

This builds a synthetic 30x40 image (a slow background gradient with
a small bright spot and a small dark spot), applies both top-hats
with a 7x7 square element, and prints the peak response at each spot
versus the near-zero response elsewhere.

Use it from Python:

```python
import numpy as np
from top_hat_transform.tophat import white_tophat, black_tophat

image = np.load("photo_gray.npy")  # 2D array
bright_spots = white_tophat(image, kernel_size=7, shape="disk")
dark_spots = black_tophat(image, kernel_size=7, shape="disk")
```

`kernel_size` must be an odd positive integer; `shape` is `"square"`
or `"disk"`.

## Run tests

```bash
python -m pytest tests/test_top_hat_transform.py -v
```
