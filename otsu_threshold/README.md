# Otsu's Thresholding, from scratch

A tiny binarization building block — no OpenCV, just numpy. It finds
the threshold that best splits a grayscale image into foreground and
background by maximising the between-class variance of the two pixel
populations.

## Why

Most document scans, cell images, and industrial inspection photos
are really two populations of pixels (ink vs paper, object vs
background) and need to become black-and-white before anything else
can be measured. Picking the threshold by hand is guesswork; Otsu's
method derives it from the histogram, so it adapts to each image.

## Usage

```bash
pip install -r requirements.txt
python -m otsu_threshold.demo
```

This builds a synthetic 40x60 bimodal image (dark pixels around 50,
bright pixels around 200), finds the Otsu threshold, and prints it
plus the foreground pixel share.

Use it on your own image from Python:

```python
import numpy as np
from otsu_threshold.otsu import otsu_threshold, binarize

image = np.load("photo_gray.npy")   # 2D array, values in [0, 255]
threshold = otsu_threshold(image)             # int in [0, 255]
mask, threshold = binarize(image)             # boolean mask, True = foreground
```

Pixels with value `<= threshold` are background, pixels `> threshold`
are foreground. A constant image has no split to find, so its single
value is returned as the threshold.

## Run tests

```bash
python -m pytest tests/test_otsu_threshold.py -v
```
