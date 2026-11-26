# pencil_sketch

Pencil-sketch effect for grayscale or RGB images, implemented with
**numpy + stdlib only** (no OpenCV).

## What it does

The classic pencil-sketch recipe:

1. Convert to grayscale (luminance weights 0.299, 0.587, 0.114).
2. Invert the grayscale image.
3. Gaussian-blur the inverted image (separable blur, replicate padding,
   kernel covering ±3σ).
4. Colour-dodge blend the grayscale image with the blurred inverted
   image: `result = base / (255 - blend) * 255`, clipped to [0, 255]
   (blend == 255 gives 255).

Flat bright areas stay white, edges come out dark — like a pencil
drawing. A flat white image stays white; a flat black image stays
black, because colour-dodge of base 0 is 0 for any blend value.

## Usage

```python
import numpy as np
from pencil_sketch import pencil_sketch

image = np.asarray(...)  # 2D grayscale or 3D RGB, any numeric dtype
sketch = pencil_sketch(image, sigma=1.0)  # uint8, same height/width
```

Other exports: `gaussian_blur(image, sigma=1.0)`,
`rgb_to_grayscale(image)`, `color_dodge_blend(base, blend)`.

`pencil_sketch` raises `TypeError` for non-array or non-numeric input
and `ValueError` for empty images, wrong shapes (not 2D grayscale or
3D RGB), or a non-positive / non-finite `sigma`.

## Demo

```bash
python pencil_sketch/demo.py
```

Builds a synthetic 60×80 image (gradient + dark disc + dark
rectangle) and prints the sketch statistics.

Demo output:

```text
input shape: (60, 80), output shape: (60, 80), dtype: uint8
output min: 62, max: 255
flat bright region mean: 255.0 (should stay near white, > 200)
edge mean: 141.6 (should be dark, clearly below the flat region)
overall mean: 250.7
```

## Tests

```bash
python -m pytest tests/test_pencil_sketch.py -q
```

11 tests, all passing: output shape/dtype/range, flat white stays
white, flat black stays black, edge darker than flat region on a step
image, grayscale and RGB inputs, blur noise reduction and shape
preservation, colour-dodge known values, invalid sigma, non-array
input, and the demo behaviour (flat region near white, edge dark).
