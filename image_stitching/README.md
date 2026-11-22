# Image Stitching

Panorama stitching from scratch (numpy only).

Two overlapping views of the same scene are aligned by searching
integer translations for the one with the highest normalized
cross-correlation (NCC) over the overlap, then placed on a canvas
covering their union and blended with a linear feather ramp across
the overlap so the seam is smooth. Only integer translations are
estimated (no rotation, scaling or perspective). Grayscale (H, W)
and colour (H, W, C) images are both supported; the panorama is
returned as a `float64` array.

## Run

```bash
python image_stitching/demo.py
python -m pytest tests/test_image_stitching.py -q
```

## Demo output

Two 64×64 views cut from one synthetic scene (gradient + shapes +
noise), img2 shifted by (dy=+5, dx=+20) relative to img1:

```
img1 shape: (64, 64), img2 shape: (64, 64)
true shift (dy, dx): (5, 20)
found shift (dy, dx): (5, 20)
shift recovered exactly: True
NCC score at best shift: 1.000000
panorama shape: (69, 84)
max |panorama - img1| in img1-only region: 0.000000
max |panorama - scene| over covered region: 0.000000
mean |step| across seam column: 11.9456
mean |step| of scene itself: 11.6327
```

The shift is recovered exactly with NCC 1.0, the panorama is the
69×84 union of the two views, the non-overlap region is copied
exactly, and the step across the seam is no larger than the scene's
own gradient — i.e. no visible seam.

## API

- `find_best_shift(img1, img2, max_shift=None)` → `((dy, dx), score)`.
  Searches candidate integer translations of img2 relative to img1
  (default `max_shift` is half the smaller image side), scores each
  by NCC over the overlap, and returns the best shift and its score.
  Ties break towards the smallest shift, so identical images align
  at (0, 0).
- `stitch_images(img1, img2, shift=None)` → panorama array. Finds
  the shift when not given, builds the union canvas, copies
  non-overlap areas exactly, and blends the overlap with a linear
  feather ramp (img2's weight ramps 0 → 1 across the overlap width,
  or across its height when the overlap is one pixel wide). Images
  placed apart still produce a valid canvas, with zeros in the gap.
- `normalized_cross_correlation(a, b)` → float in [-1, 1]; 1.0 for
  identical patches (up to a brightness change), 0.0 when either
  patch is constant.
- `overlap_region(shape1, shape2, shift)` → overlap ranges in both
  images' coordinates, or `None` when they do not overlap.

## Validation

All functions raise `ValueError` on 1D/4D or empty images, mismatched
dimensions or channel counts, non-finite values (NaN/inf), a negative
or non-integer `max_shift`, and a `shift` that is not a pair of ints.
