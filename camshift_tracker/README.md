# CamShift Tracker

CamShift (continuously adaptive mean-shift) colour tracking,
implemented from scratch with numpy only — no OpenCV.

A 3D RGB colour histogram is built from the target region, each new
frame is backprojected into a probability map (how likely each pixel's
colour is under the target model), mean-shift climbs that map to the
probability centroid, and the search window is resized from the zeroth
moment — so the tracker follows a target that moves and changes size.

## Usage

```python
import numpy as np
from camshift_tracker import build_histogram, camshift_track, track_sequence

frame = np.load("frame.npy")  # (H, W, 3), values in [0, 255]
hist = build_histogram(frame, window=(x, y, w, h), bins=16)
window, mass = camshift_track(next_frame, (x, y, w, h), hist)
windows = track_sequence(frames, (x, y, w, h), hist)
```

`build_histogram` returns a normalised histogram of shape
`(bins, bins, bins)` summing to 1. `backproject(frame, hist)` returns
the probability map in [0, 1]. `mean_shift(prob_map, window)` iterates
mean-shift on a probability map directly. `track_sequence` returns the
window after each frame.

Frames must be `(H, W, 3)` with values in [0, 255]; windows must lie
inside the frame with positive width/height; `bins` must be in [2, 64].
Anything else raises `ValueError` with a message saying what was wrong.

## Run the demo

```bash
PYTHONDONTWRITEBYTECODE=1 python camshift_tracker/demo.py
```

Builds 12 synthetic frames with a red disc moving diagonally across a
green background, initialises CamShift from a small box fully inside
the disc in frame 0, and prints the tracked centre and error per frame.

## Run the tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_camshift_tracker.py -q
```
