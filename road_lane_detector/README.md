# Road Lane Detector (Sobel + Hough, numpy only)

Detect the left and right lane lines of a road image without OpenCV:
Sobel gradient magnitude, edge thresholding, then Hough-style line
voting (an accumulator over rho/theta, restricted to near-vertical
angles) separately for the left and right halves of the image. The
strongest line in each half is returned as endpoints at the bottom of
the image and at the horizon (vanishing-point height). A synthetic
road generator (two lane lines converging to a vanishing point, dashed
centre line, optional noise) is included for demos and tests.

## Usage

```python
from road_lane_detector import detect_lanes, make_road_image

image, truth = make_road_image(noise=5.0, seed=5)
result = detect_lanes(image)
print(result["left"])   # ((x_bottom, y_bottom), (x_top, y_top)) or None
print(result["right"])
print(result["votes"])  # Hough votes per side
```

## Run demo / tests

```bash
python road_lane_detector/demo.py
python -m pytest tests/test_road_lane_detector.py -v
```

The demo generates a noisy 320x240 road, detects both lanes, and
prints the detected endpoints and their error against the true lane
geometry.
