# Blob Detection (LoG), from scratch

A tiny Laplacian-of-Gaussian blob detector — no OpenCV, just numpy.
It scans a range of scales, finds the local maxima of the
sigma-normalised LoG response across space *and* scale, and reports
each blob's centre, radius (`sqrt(2) * sigma`, the standard LoG
convention) and response strength.

## Why

Blobs are the round counterpart to edges and corners: a bright disc,
a cell nucleus, a coin on a table. A single LoG filter only fires at
one size, so scale-space detection is the natural next step after
Gaussian blur and edge detection — it shows why scale matters, and
why normalising the response by `sigma**2` is what makes peaks at
different scales comparable. The detector also shows two practical
lessons: replicate padding keeps flat regions flat right up to the
border (zero padding would invent border structure), and a sharp disc
rim fires small-scale responses of its own, so overlapping detections
are pruned in favour of the strongest blob.

## Usage

```bash
pip install -r requirements.txt
python blob_detection/demo.py
```

This builds a synthetic 100x120 image (three bright discs of radii 4,
8 and 14 on black) and prints the blobs found — all three discs, with
radii close to the true ones.

Use it on your own image from Python:

```python
import numpy as np
from blob_detection.blobs import detect_blobs

image = np.load("photo_gray.npy")   # 2D grayscale array
blobs = detect_blobs(image)         # [(row, col, radius, response), ...]
for row, col, radius, response in blobs:
    print(f"blob at ({row}, {col}), radius {radius:.1f}")
```

`min_sigma` / `max_sigma` / `num_sigma` set the scales scanned
(default 1–10, 5 scales); `threshold` is the minimum response for a
blob (absolute, so scale it to the image amplitude — for 0-255 images
the responses are ~100x larger than for 0-1 images); `overlap`
(default 0.5) prunes a blob whose circle overlaps a stronger blob's
circle by more than that fraction of the smaller circle's area
(`1.0` disables pruning).

The building blocks are exposed too:

```python
from blob_detection.blobs import log_kernel, log_response

kernel = log_kernel(sigma=2.0)        # 2D LoG kernel, sums to zero
response = log_response(image, 2.0)   # -sigma**2 * (LoG * image)
```

## Run tests

```bash
python -m pytest tests/test_blob_detection.py -v
```
