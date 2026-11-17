# hough_circles

Hough circle detection from scratch (numpy only).

Every edge pixel votes for all centres that could lie on a circle of a
given radius through it. Votes accumulate in a 3D accumulator indexed
by `(row, col, radius)`; peaks are circles. Non-maximum suppression
merges peaks closer than `min_distance` with radii within 2 of each
other, so one real circle is reported once.

## API

- `hough_circle_transform(edge_map, radii, angle_step=1.0)` — returns
  the `(height, width, len(radii))` int64 accumulator and the radii.
- `detect_circles(edge_map, radii, threshold=None, min_distance=5)` —
  returns a list of `Circle(x, y, radius, votes)`, strongest first.
  The default threshold is `max(8, int(0.5 * 2 * pi * min(radii)))`.
- `sobel_edges(gray, threshold=50.0)` — boolean edge map from a
  grayscale image (Sobel gradient magnitude).
- `circle_edge_map(shape, circles)` — synthetic edge map with circles
  drawn on it (useful for tests and demos).

## How to run

```bash
python hough_circles/demo.py
python -m pytest tests/test_hough_circles.py -q
```

## Demo output

```
edge map: (60, 80), 192 edge pixels
detected 2 circle(s):
  centre=(55, 40) radius=14 votes=112
  centre=(25, 20) radius=10 votes=80
```

The demo draws two circles — centre (25, 20) radius 10 and centre
(55, 40) radius 14 — on a 60x80 edge map and recovers both exactly.
11 tests pass.
