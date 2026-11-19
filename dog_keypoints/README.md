# Difference-of-Gaussians Keypoints, from scratch

Detects blob/corner keypoints the way the first stage of SIFT does:
blur the image at two scales, subtract the blurs, and keep the local
maxima of the absolute difference that clear a contrast threshold —
numpy only, no OpenCV.

## Why

A difference-of-Gaussians (DoG) responds strongly to structure whose
size matches the scale pair and barely at all to flat areas, so its
peaks are stable, repeatable points to anchor matching, tracking, or
recognition on. This is the same idea SIFT uses before refining each
keypoint's location and scale.

## Usage

```bash
pip install -r requirements.txt
python dog_keypoints/demo.py
```

This builds a synthetic 60x80 image (a bright disc and a bright
rectangle on black, plus noise), computes the DoG with sigma 1.0 and
1.6, and prints how many keypoints were found and where the strongest
few are.

Use it on your own image from Python:

```python
import numpy as np
from dog_keypoints.keypoints import detect_keypoints

image = np.load("photo_gray.npy")   # 2D grayscale array
keypoints = detect_keypoints(image, sigma1=1.0, sigma2=1.6,
                             contrast_threshold=5.0)
for row, col, response in keypoints:
    print(row, col, round(response, 1))
```

`sigma1`/`sigma2` set the scale pair (`sigma2` must be larger than
`sigma1`); `contrast_threshold` drops weak responses; `max_keypoints`
keeps only the strongest N. Border pixels are never keypoints.

## Run tests

```bash
python -m pytest tests/test_dog_keypoints.py -v
```
