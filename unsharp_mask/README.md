# unsharp_mask

Unsharp masking / image sharpening, implemented from scratch (numpy only).

## What is unsharp masking?

Despite the name, unsharp masking *sharpens* an image in three steps:

1. **Blur** the image with a Gaussian blur.
2. **Subtract** the blur from the original — what is left is the fine
   detail (edges and texture): `detail = image - blurred`.
3. **Add the detail back**, amplified: `sharpened = image + amount * detail`.

Flat regions have almost no detail, so they are barely touched; edges,
where the original and the blur differ most, get the biggest boost.

## Parameters

- `amount` — strength of the sharpening (`>= 0`). `0` returns the image
  unchanged; `1.0` is a typical default; larger values sharpen harder but
  can overshoot and create halos.
- `radius` — sigma of the Gaussian blur (`> 0`). It sets the scale of the
  detail that gets sharpened: a small radius sharpens fine edges, a larger
  radius sharpens broader features.
- `threshold` — minimum detail magnitude, in intensity units (`>= 0`).
  The boost is only applied where `|image - blurred| > threshold`; pixels
  below it keep their original value. A small threshold (e.g. `5`–`10`)
  keeps noise in flat regions from being amplified.

The result is clipped to `[0, 255]` and returned as a float64 array with
the same shape as the input. Both 2D grayscale and `(H, W, 3)` RGB images
are supported.

## Usage

```python
import numpy as np
from unsharp_mask import unsharp_mask, gaussian_blur

image = np.load("photo.npy")  # grayscale or RGB, any numeric dtype
sharp = unsharp_mask(image, amount=1.0, radius=1.0, threshold=5.0)
blurred = gaussian_blur(image, sigma=1.0)
```

## Run the demo

From the repo root:

```bash
python unsharp_mask/demo.py
```

It builds a synthetic 64x64 image (a sharp white square on black, plus
seeded noise) and prints the input/blurred/sharpened statistics, the boost
at a square edge, and a check that a flat region stays unchanged with a
threshold applied.

## Run the tests

From the repo root:

```bash
python -m pytest tests/test_unsharp_mask.py -q
```
