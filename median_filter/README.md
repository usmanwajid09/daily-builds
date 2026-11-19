# Median Filter, from scratch

Removes salt-and-pepper (impulse) noise by replacing each pixel with
the median of its neighbourhood — numpy only, no OpenCV.

## Why

A single white or black spike is outvoted by its neighbours, so the
median erases it — while a real step edge survives, because the median
of a window straddling an edge still sits on one side of it. A mean
filter would smear both the spike and the edge. That edge-preserving
behaviour is why the median filter is the standard cure for impulse
noise from faulty sensors or transmission errors.

## Usage

```bash
pip install -r requirements.txt
python median_filter/demo.py
```

This builds a synthetic 40x60 image (a bright disc and a bright
rectangle on black), corrupts 8% of its pixels with salt-and-pepper
noise, median-filters it with a 3x3 window, and prints the noisy-pixel
count and the mean absolute error against the clean image before and
after.

Use it on your own image from Python:

```python
import numpy as np
from median_filter.filter import median_filter

image = np.load("noisy_gray.npy")   # 2D array, or (H, W, 3) RGB
clean = median_filter(image, kernel_size=3)  # same shape, float64
```

`kernel_size` must be an odd positive integer (3 is the usual choice;
5 removes denser noise but softens fine detail). Replicate padding
keeps flat borders flat. `salt_and_pepper_pixel_count` counts the
pixels sitting exactly at 0 or 255.

## Run tests

```bash
python -m pytest tests/test_median_filter.py -v
```
