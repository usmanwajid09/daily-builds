# Region Growing Segmentation, from scratch

A tiny seeded segmentation tool — no OpenCV, just numpy. Starting
from one or more seed pixels, it flood-fills outwards over 4- or
8-connected neighbours, adding any pixel whose intensity is within a
tolerance of the seed pixel's intensity. The result is a boolean mask
of the grown region.

## Why

Region growing is the simplest interactive segmentation: instead of
thresholding the whole image at once, you point at the object you
want and let similarity to that seed decide the boundary. A small
tolerance keeps the region tight to a uniform object; a larger one
lets it follow a gentle gradient. 8-connectivity bridges diagonal
touches that 4-connectivity treats as separate.

## Usage

```bash
pip install -r requirements.txt
python region_growing/demo.py
```

This builds a synthetic 40x60 image (two bright discs on a dark
background), grows a region from a seed in the first disc, and prints
the region size (close to the disc's area) plus an ASCII mask.

Use it on your own image from Python:

```python
import numpy as np
from region_growing.region_growing import region_grow

image = np.load("photo_gray.npy")  # 2D array
mask = region_grow(image, seeds=[(100, 120)], tolerance=15.0,
                   connectivity=8)  # boolean mask
```

## Run tests

```bash
python -m pytest tests/test_region_growing.py -v
```
