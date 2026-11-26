# cartoonizer

Turn an image into a cartoon / posterized version — flat colour regions,
bold outlines, no new dependencies (numpy + stdlib only).

## How it works

1. **Edge-preserving smoothing** (`edge_preserving_smooth`) — a small
   bilateral-style filter: each pixel is a weighted average of its 5×5
   neighbourhood, with spatial and intensity-range Gaussian weights. Noise
   in flat areas is averaged away, while a sharp step edge gets ~zero
   weight across the step, so the edge survives.
2. **Colour quantization** (`quantize_colors`) — uniform per-channel bins
   (default 8 per channel), so the palette collapses to a handful of flat
   colours.
3. **Edge darkening** (`edge_mask` + `cartoonize`) — a Sobel gradient
   magnitude mask marks the outlines; those pixels are set to black.

Grayscale input is accepted and returned as RGB (channels duplicated).
Float images in [0, 255] are converted to uint8. Invalid input raises
`ValueError` / `TypeError` with a helpful message.

## Usage

```python
import numpy as np
from cartoonizer import cartoonize

image = ...  # uint8 RGB array, shape (H, W, 3)
cartoon = cartoonize(image, n_colors=8, edge_threshold=30.0,
                     smooth_iterations=1)

# Individual steps are available too:
from cartoonizer import edge_preserving_smooth, quantize_colors, edge_mask
smooth = edge_preserving_smooth(image, iterations=1, sigma_s=1.0, sigma_r=25.0)
flat = quantize_colors(image, n_colors=4)
edges = edge_mask(image, threshold=30.0)  # boolean array
```

## Demo

```
$ python cartoonizer/demo.py
Scene shape: (60, 80, 3), dtype: uint8
Distinct colours before cartoonize: 2710
Distinct colours after cartoonize:  8
Edge pixels (edge_mask, threshold=30): 423 of 4800
Cartoon output shape: (60, 80, 3), dtype: uint8
```

The synthetic scene is a 60×80 sky with a red rectangle, a green disc and
a blue block, plus a little noise. Cartoonizing collapses its 2710 noisy
colours down to 8 flat colours, with 423 edge pixels drawn in black.

## Tests

```
$ python -m pytest tests/test_cartoonizer.py -q
12 passed
```

Covers output shape/dtype, colour reduction, quantization limits,
edge-mask behaviour on step vs. flat images, noise reduction with step
preservation, invalid inputs, grayscale and float input, and the demo
scene's colour reduction.
