# Laplacian Pyramid, from scratch

Builds Gaussian and Laplacian image pyramids with numpy only — no
OpenCV — and uses them to reconstruct an image losslessly and to blend
two images seamlessly across scales.

## How pyramids work

A **Gaussian pyramid** is a stack of the same image at smaller and
smaller sizes: each level is blurred (5x5 binomial kernel
`[1, 4, 6, 4, 1] / 16`) and downsampled by 2, so each level is half the
width and height of the one above it.

A **Laplacian pyramid** stores the detail lost between successive
Gaussian levels: level *i* is Gaussian level *i* minus the upsampled
Gaussian level *i+1*, and the smallest level is kept whole. Adding the
levels back — expand and add, from the smallest up — rebuilds the
original image exactly (up to floating-point error).

That multi-scale split is what makes **blending** work: split both
images into Laplacian pyramids and the mask into a Gaussian pyramid,
combine them level by level (`A * mask + B * (1 - mask)`), and
reconstruct. Coarse levels blend over a wide area, fine levels only
near the seam, so the join is invisible at every scale.

## Usage

```bash
python -m laplacian_pyramid.demo
```

This builds a synthetic 64x64 image (gradient + disc + square), builds
a 4-level Laplacian pyramid, reconstructs it, and prints the level
shapes and the (near-zero) reconstruction error, then blends two images
with a half/half mask.

Use it on your own image from Python:

```python
import numpy as np
from laplacian_pyramid.pyramid import (
    build_laplacian_pyramid, reconstruct, blend_images,
)

image = np.load("photo_gray.npy")        # 2D grayscale, or (H, W, 3) RGB
pyramid = build_laplacian_pyramid(image, levels=4)
rebuilt = reconstruct(pyramid)           # same shape, ~exact copy

blended = blend_images(image_a, image_b, mask, levels=4)
# mask: 2D float array, same H/W, 1 = take A, 0 = take B
```

`reduce` blurs and downsamples by 2; `expand` upsamples by 2
(zero-insertion + blur scaled x4 to preserve energy). `levels` must be
>= 1 and small enough for the image size (the smallest level must stay
at least 1 pixel). Inputs may be grayscale 2D or RGB 3D; invalid shapes
or levels raise `ValueError`.
