# Skin Detector (YCbCr rule-based segmentation)

Segment skin pixels in an RGB image using the classic rule-based test
in YCbCr colour space: a pixel is skin when Cb is in [77, 127] and Cr
is in [133, 173] (Chai & Ngan bounds). Skin clusters tightly in the
chrominance plane, so this needs no training data. The module also
reports the skin fraction and the bounding box of the largest connected
skin region.

## Why

YCbCr skin segmentation is a standard first step in face detection,
hand tracking, and gesture recognition pipelines. It shows how a
colour-space change turns a hard RGB thresholding problem into two
simple range checks.

## Usage

```bash
pip install -r requirements.txt
python -m skin_detector.demo
```

The demo builds a synthetic scene (blue background, green circle, one
skin-coloured rectangle) and prints the skin fraction, pixel count, and
largest-region bounding box, which should match the rectangle exactly.

```python
from skin_detector import detect_skin, skin_stats

mask = detect_skin(image)          # boolean (H, W) mask of skin pixels
print(skin_stats(image))           # fraction, counts, largest bbox
```

## Run tests

```bash
python -m pytest tests/test_skin_detector.py -v
```
