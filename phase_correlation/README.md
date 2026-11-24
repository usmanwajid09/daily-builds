# Phase Correlation

FFT-based translation estimation between two images, implemented from
scratch with numpy only — no OpenCV.

The normalised cross-power spectrum of the two images is inverse
Fourier transformed; its peak sits at the translation between them.
Because only phase is used, the estimate is insensitive to uniform
brightness changes, and a parabolic fit around the peak refines the
shift to sub-pixel accuracy.

## Usage

```python
import numpy as np
from phase_correlation import phase_correlate, estimate_translation

dy, dx, peak = phase_correlate(image1, image2)
result = estimate_translation(image1, image2)
result["shift"]      # (dy, dx): how image2 is shifted vs image1
result["peak"]       # correlation peak in [0, 1]; ~1 for a pure shift
result["magnitude"]  # sqrt(dy**2 + dx**2)
```

Positive `dy`/`dx` mean image2 is shifted down/right relative to
image1. `shift_image_fourier(image, dy, dx)` shifts an image by a
(fractional) amount via the Fourier shift theorem — useful for making
test pairs. Images must be 2D grayscale, the same shape, and
non-constant (a flat image carries no phase information); anything
else raises `ValueError`.

## Run the demo

```bash
PYTHONDONTWRITEBYTECODE=1 python phase_correlation/demo.py
```

Builds a synthetic 64x64 image, shifts it by (dy=+3.5, dx=-2.25)
pixels, and prints the estimated shift and the error.

## Run the tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_phase_correlation.py -q
```
