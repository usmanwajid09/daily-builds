# Shi-Tomasi Corner Detector, from scratch

A tiny corner detector — no OpenCV, just numpy. It builds the
structure tensor from Sobel gradients (summed over a small window),
takes the smaller eigenvalue at each pixel as the corner response,
and returns the strongest corners after non-maximum suppression by a
minimum distance. Flat regions and straight edges score near zero;
true corners score high.

## Why

Shi-Tomasi (1994) improves on Harris by using the smaller eigenvalue
directly instead of a determinant-minus-trace heuristic, which makes
the response easier to interpret: a pixel is a corner exactly when
the image varies strongly in *both* directions around it. It is the
`goodFeaturesToTrack` criterion used by OpenCV.

## Usage

```bash
pip install -r requirements.txt
python shi_tomasi_corners/demo.py
```

This builds a synthetic 30x30 image (a bright square on a dark
background), computes the corner response, and prints the four
detected corners — the square's corners — with their strengths.

Use it on your own image from Python:

```python
import numpy as np
from shi_tomasi_corners.shi_tomasi_corners import detect_corners

image = np.load("photo_gray.npy")  # 2D array
corners = detect_corners(image, max_corners=50, quality_level=0.01,
                         min_distance=5)
for row, col, strength in corners:
    print(row, col, strength)
```

## Run tests

```bash
python -m pytest tests/test_shi_tomasi_corners.py -v
```
