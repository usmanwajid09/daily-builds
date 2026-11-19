# contour_approximation

Contour approximation: Moore-Neighbour tracing + Douglas-Peucker
simplification (numpy only).

The outer boundary of the foreground region is traced with
Moore-Neighbour tracing (the boundary walk keeps the foreground on
one side and stops by Jacob's criterion when it returns to the start
pixel), then simplified with the Douglas-Peucker algorithm into a
polygon with far fewer vertices. The result reports the simplified
vertices, the polygon area (shoelace formula) and the perimeter.
Only the region containing the topmost-leftmost foreground pixel is
traced.

## API

- `trace_contour(binary)` — returns an `(N, 2)` int array of
  `(x, y)` boundary points in trace order (start pixel not repeated).
- `douglas_peucker(points, epsilon)` — simplifies an open or closed
  polyline; returns a float array with the first point repeated at
  the end for closed input.
- `approximate_contour(binary, epsilon=1.0)` — returns a dict with
  `vertices` (closed), `area`, `perimeter` and `raw_points`.
- `polygon_area(points)`, `polygon_perimeter(points)` — shoelace area
  and closed-polygon perimeter helpers.

## How to run

```bash
python contour_approximation/demo.py
python -m pytest tests/test_contour_approximation.py -q
```

## Demo output

```text
binary image: (50, 70), 397 foreground pixels
traced contour: 56 raw boundary points
  epsilon=0.5: 4 vertices, area=171.0, perimeter=56.0
    vertices: [[20, 10], [39, 10], [39, 19], [20, 19]]
  epsilon=2.0: 4 vertices, area=171.0, perimeter=56.0
    vertices: [[20, 10], [39, 10], [39, 19], [20, 19]]
```

The traced region is the filled 10x20 rectangle, whose 56 boundary
pixels collapse to its 4 corners (pixel-centre area 171.0,
perimeter 56.0) at either tolerance. 20 tests pass.
