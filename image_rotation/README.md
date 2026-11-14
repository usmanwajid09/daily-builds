# Image Rotation, from scratch

Rotate a grayscale or RGB image by any angle — no OpenCV, just numpy.
Two interpolation methods: nearest-neighbour and bilinear. The rotation
is done by inverse mapping: every output pixel samples the source image
at the location you get by rotating its offset from the centre back by
the angle, so there are no holes in the output.

## Why

Rotation is the classic first step into geometric image transforms: it
shows why you map backwards (output → source) instead of forwards, how
interpolation decides what a pixel "between" two source pixels looks
like, and what happens at the borders when the rotated image no longer
fits its frame.

## Same-size canvas — read this first

The output has **exactly the same shape as the input**; the canvas is
not enlarged. That means for large angles the corners of the rotated
image are cropped off, and the vacated corners are filled with `fill`
(default `0.0`). This is the documented behaviour, not a bug. If you
need the whole rotated image, pad the input first (e.g. with
`np.pad`) and crop afterwards.

The fill value only ever applies to output pixels whose source location
falls **outside** the image. A source point inside the image always
interpolates from real pixels only — the fill never leaks in through a
neighbour that happens to sit just outside the border. So a uniform
image stays uniform wherever the source is in range, at any angle.

## Usage

```bash
pip install -r requirements.txt
python -m image_rotation.demo
```

This builds a synthetic 21x21 image with a bright diagonal line,
rotates it 45 degrees counter-clockwise with both methods, and prints
pixel sums plus an ASCII rendering before and after — the diagonal
turns into a horizontal line.

Use it on your own image from Python:

```python
import numpy as np
from image_rotation.rotate import rotate_image

image = np.load("photo_gray.npy")   # 2D array, or (H, W, 3) RGB
rotated = rotate_image(image, 45)                # bilinear, fill 0.0
rotated = rotate_image(image, -30, method="nearest", fill=255.0)
```

`angle` is in degrees and counter-clockwise is positive; `method` is
`"bilinear"` (default, smooth) or `"nearest"` (crisp, no blending);
`fill` is the value for pixels that map outside the source image.

## Run tests

```bash
python -m pytest tests/test_image_rotation.py -v
```
