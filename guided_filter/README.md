# Guided Filter, from scratch

A tiny edge-preserving smoother — no OpenCV, just numpy. It
implements the guided filter of He, Sun and Tang (2013): in each
local window the output is a linear function of a guide image,
`q = a*I + b`, with `a` and `b` fitted by ridge regression of the
input on the guide inside that window. Averaging the per-window
models gives the output. Where the guide is flat, `a` is near zero
and the window mean wins (strong smoothing); across a sharp guide
edge, `a` is near one and the edge passes through untouched.

## Why

Unlike a plain blur, which softens everything, the guided filter
knows where the edges are (from the guide) and refuses to smooth
across them — so it denoises flat areas while keeping boundaries
crisp. With the image itself as the guide it is a drop-in
edge-preserving smoother; with a different guide (e.g. a clean
reference) it can transfer that guide's structure onto a noisy input.

## Usage

```bash
pip install -r requirements.txt
python guided_filter/demo.py
```

This builds a synthetic 40x60 image (a step edge plus Gaussian
noise), smooths it, and prints the noise level in a flat region
before/after and how much of the step height survives.

Use it on your own image from Python:

```python
import numpy as np
from guided_filter.guided_filter import edge_preserving_smooth, guided_filter

image = np.load("photo_gray.npy")  # 2D array
smooth = edge_preserving_smooth(image, radius=4, eps=100.0)
# or filter a noisy image using a clean guide:
smooth = guided_filter(guide_image, noisy_image, radius=4, eps=0.1)
```

## Run tests

```bash
python -m pytest tests/test_guided_filter.py -v
```
