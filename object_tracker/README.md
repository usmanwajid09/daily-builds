# Object Tracker (NCC template tracking)

Track an object across frames by normalised cross-correlation (NCC):
cut a template patch from the first frame, then for each new frame
search a window around the last known position for the patch most
similar to the template. The tracker reports the bounding box and the
NCC score each frame, flags frames that fall below a minimum score as
lost, and `mean_drift` measures the average distance (px) between the
tracked centres and a ground-truth path.

## Usage

```python
from object_tracker import ObjectTracker, make_sequence, mean_drift

frames, true_centres = make_sequence(n_frames=20, start=(30, 30),
                                     velocity=(3, 2), radius=10)
tracker = ObjectTracker(search_radius=15).init(frames[0], (20, 20, 21, 21))
results = tracker.track_sequence(frames[1:])  # list of (bbox, score)
tracked = [(20, 20, 21, 21)] + [b for b, _ in results]
print(mean_drift(tracked, true_centres))  # mean drift in px
```

## Run demo / tests

```bash
python object_tracker/demo.py
python -m pytest tests/test_object_tracker.py -v
```

The demo tracks a disc moving at (3, 2) px/frame over 20 noisy
frames and prints the mean drift, final position error and worst NCC
score.
