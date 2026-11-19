# convex_hull

Convex hull via Graham scan, with a Jarvis march cross-check
(numpy only).

Two classic algorithms are implemented independently: the Graham
scan (sort by polar angle around the lowest point, keep only left
turns, O(n log n)) and the Jarvis march / gift wrapping (repeatedly
pick the most counter-clockwise next point, O(n h)). Both return
the hull vertices counter-clockwise without repeating the start
vertex, keeping only extreme endpoints on collinear edges. Also
provided: `polygon_area` (shoelace formula) and `point_in_hull`
(a point is inside a counter-clockwise convex polygon exactly when
it lies left of every edge).

## API

- `graham_scan(points)` — `(H, 2)` hull vertices, counter-clockwise.
- `jarvis_march(points)` — same contract, independent algorithm.
- `polygon_area(points)` — shoelace area (0.0 for fewer than 3 points).
- `point_in_hull(point, hull)` — True inside or on the boundary;
  handles 1- and 2-point degenerate hulls.

## How to run

```bash
python convex_hull/demo.py
python -m pytest tests/test_convex_hull.py -q
```

## Demo output

```text
30 random points in a 10x10 square (seed 42)
Graham scan hull: 8 vertices
  vertices: [[2.0, 0.07], [4.67, 0.44], [5.55, 0.64], [7.78, 1.95], [9.71, 8.93], [7.45, 9.68], [0.94, 9.76], [1.4, 1.15]]
Jarvis march hull: 8 vertices, same set: True
hull area: 66.87 (square is 100.00)
centroid [5.27 5.13] inside hull: True
point (20, 20) inside hull: False
```

19 tests pass.
