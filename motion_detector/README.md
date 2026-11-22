# motion_detector

Frame-differencing motion detector from scratch — numpy only, no
OpenCV, no scipy.

## What it does

`MotionDetector` keeps a running background model, updated as an
exponential moving average of the frames seen so far:

```
background = (1 - lr) * background + lr * frame
```

A pixel counts as motion when its absolute difference from the
background exceeds `threshold`. Motion pixels are grouped into regions
with a from-scratch two-pass **union-find connected-components**
labelling (8-connectivity), and each region whose pixel area is at
least `min_area` is reported as a bounding box. Smaller regions
(sensor noise, a few isolated changed pixels) are filtered out.

## Input convention

Frames are 2D grayscale arrays. Integer arrays (e.g. `uint8`) are
interpreted on the 0–255 scale and divided by 255; float arrays are
interpreted on the 0–1 scale and must lie in [0, 1]. Everything is
normalised internally to float64 on the 0–1 scale, so `threshold` is
also on the 0–1 scale (default `0.1`, about 25 grey levels on 0–255).

## API

```python
from motion_detector import MotionDetector

detector = MotionDetector(threshold=0.1, learning_rate=0.05, min_area=20)
result = detector.detect(frame)
# result = {
#   "mask":            bool array, same shape as the frame,
#   "bounding_boxes":  [(y_min, x_min, y_max, x_max), ...]  # inclusive bounds
#   "motion_pixels":   int,
#   "motion_fraction": float,
# }
detector.update_background(frame)  # fold a frame into the background
detector.reset()                   # forget the background
```

Also exposed as pure functions: `detect_motion(frame, background,
threshold=0.1, min_area=20)` and `connected_components(mask)`.

Invalid input raises `ValueError` with a clear message: non-2D or
empty frames, NaN/inf values, float values outside [0, 1], a frame
shape that differs from earlier frames, `threshold < 0`,
`learning_rate` outside [0, 1], or `min_area < 1`.

## How to run

From the repo root:

```bash
python motion_detector/demo.py
```

To run the tests:

```bash
python -m pytest tests/test_motion_detector.py -q
```

## Demo

Synthetic 64×64 video: 5 static background frames (value 0.2), then
5 frames where a bright 10×10 square (value 1.0) moves +4 px to the
right each frame. The demo uses `learning_rate=0.01` — the square
overlaps its previous position by 6 columns each step, and a faster
update would leave a faint "ghost" of the old position in the
background.

### Demo output

```
frame | boxes | motion_pixels | box (y_min,x_min,y_max,x_max) | centre (y,x) vs true
------+-------+---------------+--------------------------------+----------------------
    0 |     0 |             0 | static frame
    1 |     0 |             0 | static frame
    2 |     0 |             0 | static frame
    3 |     0 |             0 | static frame
    4 |     0 |             0 | static frame
    5 |     1 |           100 | (27, 8, 36, 17) | (31.5, 12.5) vs (31.5, 12.5)
    6 |     1 |           100 | (27, 12, 36, 21) | (31.5, 16.5) vs (31.5, 16.5)
    7 |     1 |           100 | (27, 16, 36, 25) | (31.5, 20.5) vs (31.5, 20.5)
    8 |     1 |           100 | (27, 20, 36, 29) | (31.5, 24.5) vs (31.5, 24.5)
    9 |     1 |           100 | (27, 24, 36, 33) | (31.5, 28.5) vs (31.5, 28.5)

Static frames: 0 boxes, 0 motion pixels total (expected 0 / 0 after warm-up).
Moving frames: [1, 1, 1, 1, 1] boxes per frame (expected [1, 1, 1, 1, 1]).
Square area: 100 px; motion pixels on moving frames should be ~100 each.
```

The detected box centre matches the square's true centre exactly on
every moving frame, and the motion pixel count is exactly the square's
area (100 px).

## Tests

18 tests in `tests/test_motion_detector.py`, all passing — static
frames give no boxes, a moving square is found with exactly one box
tracking its centre, motion pixel count matches the square area,
small noise is filtered by `min_area`, threshold and learning-rate
effects, all invalid inputs raise `ValueError`, mask shape/dtype,
`reset()`, and determinism.
