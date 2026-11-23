# Colour Space Converter, from scratch

Colour-space conversions with no OpenCV, just numpy: RGB to/from HSV,
RGB to/from YCbCr (BT.601 full-range), RGB to grayscale (BT.601 luma),
and grayscale back to RGB.

## Conventions

- RGB channels are float64 in [0, 255] (uint8 input is accepted too).
- HSV: H in [0, 360) degrees, S and V in [0, 1].
- YCbCr: BT.601 full-range, Y/Cb/Cr all in [0, 255] (white maps to
  Y=255, Cb=128, Cr=128).
- Grayscale: BT.601 luma, in [0, 255].

## Why

Different tasks want different colour spaces: HSV separates hue from
brightness (handy for colour-based segmentation), YCbCr separates
luma from chroma (the basis of JPEG and video compression), and
grayscale is the input to most classical CV. Writing the conversions
by hand shows exactly what each channel means — and the round-trip
error (RGB -> HSV -> RGB, RGB -> YCbCr -> RGB) proves they are
inverses of each other.

## Usage

```bash
pip install -r requirements.txt
python color_space_converter/demo.py
```

This builds a small synthetic RGB image (red/green/blue/white patches
plus a smooth gradient), converts it to HSV, YCbCr and grayscale,
converts back, and prints the round-trip error for each conversion.

Use it from Python:

```python
import numpy as np
from color_space_converter.color_space import rgb_to_hsv, hsv_to_rgb

image = np.load("photo_rgb.npy")  # (H, W, 3), values in [0, 255]
hsv = rgb_to_hsv(image)           # H in [0, 360), S and V in [0, 1]
back = hsv_to_rgb(hsv)            # back to RGB in [0, 255]
```

## Run tests

```bash
python -m pytest tests/test_color_space_converter.py -v
```
