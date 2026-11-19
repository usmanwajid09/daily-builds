# Chroma Key (Green Screen), from scratch

Composites a foreground shot against a green screen over a background
image: pixels close to the key colour in RGB space are replaced by the
background, everything else is kept — numpy only, no OpenCV.

## Why

Chroma keying is how weather presenters, streamers, and film sets put a
person in front of any scene. The core of it is just a colour-distance
test, and doing it by hand shows the two things that make or break a
key: the tolerance (how close a colour must be to count as screen) and
the feather (a soft blend band so the cut-out edge does not look pasted
on).

## Usage

```bash
pip install -r requirements.txt
python chroma_key/demo.py
```

This builds a synthetic 60x80 foreground (a red disc on a green
screen) and a blue gradient background, composites them, and prints how
many pixels were replaced plus sample colours before and after.

Use it on your own images from Python:

```python
import numpy as np
from chroma_key.chroma import composite

foreground = np.load("shot.npy")       # (H, W, 3) RGB, green screen
background = np.load("beach.npy")      # (H, W, 3) RGB, same shape
result = composite(foreground, background, key_color=(0, 255, 0),
                   tolerance=60.0, feather=10.0)
```

`key_color` is the screen colour (green by default); `tolerance` is
the RGB distance below which a pixel counts as screen; `feather`
blends the edge over that many extra distance units. Both images must
be (H, W, 3) with the same shape. `green_mask` returns the foreground
alpha map (1 = keep, 0 = replace) and `replaced_pixel_count` counts the
screen pixels.

## Run tests

```bash
python -m pytest tests/test_chroma_key.py -v
```
