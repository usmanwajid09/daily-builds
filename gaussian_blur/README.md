# Gaussian Blur, from scratch

A tiny image-smoothing building block — no OpenCV, just numpy. It
blurs a grayscale (or RGB) image with a Gaussian kernel, applied as two
separable 1D passes (rows, then columns) with replicate padding, so
flat areas stay flat right up to the border.

## Why

Blurring is the standard first step before edge detection or corner
detection: it knocks down sensor noise so the gradients that follow
respond to real structure, not speckles. Doing it separably shows why
a Gaussian is cheap — two 1D passes of length *k* instead of one 2D
pass of *k²* — and why replicate padding matters at the border.

## Usage

```bash
pip install -r requirements.txt
python -m gaussian_blur.demo
```

This builds a synthetic 40x60 image (a sharp vertical edge plus
noise), blurs it with sigma 1.5, and prints pixel statistics before
and after, plus the edge profile so you can see the step turn into a
smooth ramp.

Use it on your own image from Python:

```python
import numpy as np
from gaussian_blur.blur import gaussian_blur

image = np.load("photo_gray.npy")   # 2D array, or (H, W, 3) RGB
blurred = gaussian_blur(image, sigma=1.5)  # same shape, float64
```

`sigma` controls the blur strength (larger = smoother); `kernel_size`
is the odd kernel length and defaults to `2 * ceil(3 * sigma) + 1`,
covering ±3 sigma.

## Run tests

```bash
python -m pytest tests/test_gaussian_blur.py -v
```
