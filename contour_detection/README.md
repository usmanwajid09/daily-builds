# contour_detection

Contour tracing from scratch (numpy only).

Foreground regions in a binary image are labelled with an 8-connected
flood fill, then each region's outer boundary is traced with
Moore-Neighbour tracing (Jacob's stopping criterion). Each contour
reports its points, polygon area (shoelace formula through pixel
centres), perimeter (sum of step lengths) and axis-aligned bounding
box. Only outer contours are returned; holes are not traced separately.

## API

- `find_contours(binary)` — returns a list of `Contour(points, area,
  perimeter, bbox)`, largest area first. `points` is an `(N, 2)` int
  array of `(x, y)` coordinates in trace order, closed by repeating
  the start pixel at the end.
- `contour_area(points)` — shoelace area (0.0 for fewer than 3 points).
- `contour_perimeter(points)` — closed-polygon perimeter (0.0 for
  fewer than 2 points).
- `bounding_box(points)` — `(x_min, y_min, x_max, y_max)`.

## How to run

```bash
python contour_detection/demo.py
python -m pytest tests/test_contour_detection.py -q
```

## Demo output

```
binary image: (50, 70), 397 foreground pixels
found 2 contour(s):
  contour 0: 45 points, area=174.0, perimeter=52.3, bbox=(42, 30)-(58, 46)
  contour 1: 57 points, area=171.0, perimeter=56.0, bbox=(20, 10)-(39, 19)
```

The demo image holds a filled 10x20 rectangle (area 171.0 through
pixel centres, perimeter 56.0) and a filled disc of radius 8
(area 174.0, perimeter 52.3). 11 tests pass.
