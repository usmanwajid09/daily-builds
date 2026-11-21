# Lucas-Kanade Tracker

Sparse optical-flow point tracking from scratch (numpy only): for
each point, the Lucas-Kanade least-squares solution over a window —
solve `(A^T A) d = A^T b` with `A = [Ix, Iy]`, `b = -It` — refined
by iterative warping with bilinear resampling. Points in flat
regions (smallest eigenvalue of `A^T A` below threshold) or outside
the frame are reported lost (status 0) instead of returning a
meaningless displacement. Points are `(y, x)` row/column coordinates.

## Run

```bash
python lucas_kanade_tracker/demo.py
python -m pytest tests/test_lucas_kanade_tracker.py -q
```

## Example

```python
from lucas_kanade_tracker import track_points

new_points, status, errors = track_points(prev, curr, points)
# new_points: (n, 2) (y, x) in curr; status: 1 tracked, 0 lost
# errors: mean absolute window residual per point
```

Demo on a synthetic textured disc moving by (dy=+2, dx=+3):

```
true shift: dy=2, dx=3
tracked points: 8/8, mean displacement: dy=2.00, dx=3.00
mean displacement error: 0.000 px
mean residual in window: 0.0002
flat-background points lost as expected: 3/3
```

## API

- `track_points(prev, curr, points, window_size=15, max_iterations=10, min_eigenvalue=1e-3)` → `(new_points, status, errors)`
- `LucasKanadeTracker(window_size=15, ...)` — stateful wrapper with `init(frame, points)` and `update(prev, curr)`
- `textured_disc_frame(shape=(64, 64), center=(32, 32), radius=10, seed=0)` — synthetic test frame: a disc filled with fixed random texture
