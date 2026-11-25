# Bayer Demosaicing, from scratch

A tiny colour-filter-array demo — no OpenCV, just numpy. A camera
sensor measures only one colour per pixel (the Bayer pattern: red and
blue on a 2-pixel grid, green on a checkerboard), so the other two
colours at each pixel have to be interpolated. This module does that
with bilinear interpolation: a missing green value is the mean of its
four orthogonal green neighbours, and a missing red/blue value is
the mean of its same-colour neighbours (left/right, up/down, or the
four diagonals, depending on where it sits). It also includes
`mosaic` (RGB -> Bayer, the inverse operation, handy for testing)
and a PSNR helper.

## Why

Demosaicing is the first step of every camera's image pipeline, and
bilinear interpolation is the baseline every fancier method (edge-
directed, gradient-corrected) is measured against. It is also a neat
example of normalised convolution: convolve the known samples with a
kernel, convolve the sample mask with the same kernel, and divide.

## Usage

```bash
pip install -r requirements.txt
python demosaicing/demo.py
```

This builds a smooth 40x60 RGB image, samples it to an RGGB Bayer
mosaic, demosaics it back, and prints the PSNR and worst per-channel
error.

Use it on your own data from Python:

```python
import numpy as np
from demosaicing.demosaicing import demosaic, mosaic, psnr

bayer = np.load("raw_bayer.npy")  # 2D array, even height and width
rgb = demosaic(bayer, pattern="RGGB")  # HxWx3 float64
print(psnr(original_rgb, rgb))  # if you have the ground truth
```

Supported patterns: `RGGB`, `BGGR`, `GRBG`, `GBRG`.

## Run tests

```bash
python -m pytest tests/test_demosaicing.py -v
```
