# image_watermark — Visible + Invisible Image Watermarking

Watermark uint8 images (grayscale or RGB) with NumPy only — no OpenCV,
no extra dependencies. Two complementary techniques:

- **Visible watermark** — a white checkerboard block pattern blended
  into one corner of the image at a chosen opacity.
- **Invisible watermark** — a key-derived bit pattern hidden in the
  least-significant bits of the pixels, verifiable later without
  visibly changing the image.

## How it works

### Visible watermark — `add_visible_watermark(image, alpha=0.3, position="bottom-right", block_size=8)`

A checkerboard of `block_size` × `block_size` squares (white squares
alternating with untouched squares) is tiled over a corner region
`4 * block_size` pixels square (32×32 by default, clipped to the image)
and blended towards white at strength `alpha`. Only the white squares
change the image, so the pattern reads as a checkerboard rather than a
solid patch. `position` is one of `"bottom-right"`, `"bottom-left"`,
`"top-right"`, `"top-left"`. Returns a new uint8 array; the input is
unchanged and pixels outside the corner are copied exactly.

### Invisible watermark — `embed_invisible_watermark(image, key)` / `verify_invisible_watermark(image, key)`

The key is hashed with SHA-256 (`hashlib.sha256`) and, prefixed by the
2-byte magic marker `b"WM"`, written into the least-significant bits of
the pixels in row-major order. The 272-bit pattern (16 magic + 256 digest
bits) is repeated as many whole times as fit in the image. Changing an
LSB moves a pixel by at most 1, so the result is visually identical
(PSNR typically well above 40 dB). `verify_invisible_watermark` returns
`True` only when the pattern for that exact key is present — `False`
for a wrong key or an unwatermarked image. Images with fewer than 272
pixels cannot hold one copy and raise `ValueError` on embed.

### Quality measure — `psnr(image_a, image_b)`

Peak signal-to-noise ratio in dB between two uint8 images of the same
shape; `inf` for identical images, `ValueError` on a shape mismatch.

## Usage

```python
import numpy as np
from image_watermark import (
    add_visible_watermark,
    embed_invisible_watermark,
    verify_invisible_watermark,
    psnr,
)

image = np.zeros((64, 64, 3), dtype=np.uint8)  # any uint8 RGB/grayscale image

# Visible: white checkerboard in the bottom-right corner, 30% opacity.
visible = add_visible_watermark(image, alpha=0.3, position="bottom-right")

# Invisible: hide a key-derived pattern in the LSBs.
hidden = embed_invisible_watermark(image, "daily-builds")
print(psnr(image, hidden))                              # ~51 dB
print(verify_invisible_watermark(hidden, "daily-builds"))  # True
print(verify_invisible_watermark(hidden, "wrong-key"))     # False
```

## Demo

Run from the repo root:

```
python image_watermark/demo.py
```

Demo output (seeded 64×64 RGB gradient image):

```
image: 64x64 RGB, dtype=uint8
visible watermark (alpha=0.3, bottom-right, block_size=8):
  mean |change| in watermarked corner: 11.98
  mean |change| elsewhere:              0.00
invisible watermark (key='daily-builds'):
  PSNR vs original: 51.12 dB
  max |pixel change|: 1
  verify correct key 'daily-builds': True
  verify wrong key 'wrong-key': False
  verify unwatermarked image: False
```

The visible watermark changes only the chosen corner (mean |change|
≈ 12.0 there, exactly 0 elsewhere); the invisible watermark leaves the
image visually untouched at ~51 dB PSNR while still verifying.

## Tests

```
python -m pytest tests/test_image_watermark.py -q
```

16 tests cover: corner-only visible changes, alpha strength, invalid
position/alpha, embed + verify (correct/wrong/unwatermarked), PSNR
> 40 dB and max |diff| ≤ 1 for the invisible watermark, too-small
images, `psnr` identical/inf and shape mismatch, empty/non-string keys,
and non-uint8 inputs.
