# Mean-Shift Tracker

Colour-histogram object tracking from scratch (numpy only). The
target is modelled by a normalised RGB histogram from the initial
bounding box; each new frame back-projects the model into per-pixel
weights `sqrt(q[u] / p[u])` (target vs. current-window histogram for
the pixel's colour bin) and shifts the window centre to the weighted
mean of the pixel positions — the mean-shift step — until the centre
barely moves. Pixels whose colour never occurred in the target get
weight zero, so the window is pulled towards the target and away
from differently coloured background.

## Run

```bash
python mean_shift_tracker/demo.py
python -m pytest tests/test_mean_shift_tracker.py -q
```

## Example

```python
from mean_shift_tracker import MeanShiftTracker

tracker = MeanShiftTracker()
tracker.init(first_frame, (y, x, height, width))
bbox = tracker.update(next_frame)  # (y, x, height, width)
```

Demo on a synthetic red square moving (dy=+2, dx=+3) per frame:

```
frame 0: true centre=(30, 40), tracked centre=(30.50, 40.50), error=0.71 px
frame 1: true centre=(32, 43), tracked centre=(32.00, 43.00), error=0.00 px
frame 2: true centre=(34, 46), tracked centre=(34.00, 46.00), error=0.00 px
frame 3: true centre=(36, 49), tracked centre=(36.00, 48.50), error=0.50 px
frame 4: true centre=(38, 52), tracked centre=(38.00, 52.00), error=0.00 px
frame 5: true centre=(40, 55), tracked centre=(40.00, 55.00), error=0.00 px
max centre error over 5 frames: 0.71 px
```

(The 0.71 px on frame 0 is the half-pixel difference between a
box's continuous centre (`y + h/2`) and the centroid of its pixels;
the tracker follows the pixels. Even-sized targets can lag a little
more — the integer window cannot contain the whole target once it is
offset by half a pixel — so odd-sized targets track best.)

## API

- `MeanShiftTracker(bins=8, max_iterations=20, epsilon=0.5)` — `init(frame, bbox)`, `update(frame)` → `(y, x, height, width)`; `.bbox` and `.center_` hold the current state
- `color_histogram(frame, bins=8)` → normalised RGB histogram, length `bins**3`
- `mean_shift_step(frame, center, size, target_hist, bins=8)` → new `(y, x)` centre after one step
- `moving_square_frame(shape=(60, 80), center=(30, 40), size=12, seed=0)` — synthetic test frame: a solid red square on a noisy dark-grey background
