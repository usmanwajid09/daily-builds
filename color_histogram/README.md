# Colour Histograms

A tiny colour-histogram toolkit — no OpenCV, just numpy. It computes
per-channel histograms for RGB images, compares histograms, and
backprojects a histogram onto an image to find pixels that match a
target colour.

## Why

A histogram summarises *what colours* an image contains while throwing
away *where* they are, which makes it a cheap, robust first step for
image retrieval ("find photos with similar colours"), tracking ("find
the pixels that look like this object") and simple scene comparison.
Comparing histograms needs no alignment between the images, and
backprojection turns a colour model into a probability map over the
whole image in one pass.

## Usage

```bash
pip install -r requirements.txt
python -m color_histogram.demo
```

The demo builds a red-dominant and a blue-dominant synthetic image,
compares their red-channel histograms with all three methods, and
backprojects the red histogram onto both images.

Use it on your own image from Python:

```python
import numpy as np
from color_histogram.hist import (
    channel_histograms, compare_histograms, backproject,
)

image = np.load("photo_rgb.npy")          # (H, W, 3) RGB array
hists = channel_histograms(image)         # list of 3 arrays, each sums to 1
red_hist = hists[0]

other = np.load("other_rgb.npy")
other_hist = channel_histograms(other)[0]
print(compare_histograms(red_hist, other_hist, "correlation"))   # ~1 = alike
print(compare_histograms(red_hist, other_hist, "chi_square"))    # ~0 = alike
print(compare_histograms(red_hist, other_hist, "intersection"))  # ~1 = alike

prob = backproject(image, red_hist)       # (H, W) map, high = looks red
```

`bins` sets the histogram resolution (default 256) and `value_range`
the intensity range covered (default `(0, 255)`). Comparing a
histogram with itself gives correlation 1.0, chi-square 0.0 and
intersection 1.0. Backprojection looks each pixel's intensity up in
the histogram on one channel (`channel=0` for red by default, `1` for
green, `2` for blue) — a single 1D histogram can only model one
channel, so pass the histogram of the channel you are tracking.

## Run tests

```bash
python -m pytest tests/test_color_histogram.py -v
```
