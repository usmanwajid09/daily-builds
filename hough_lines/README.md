# Hough Lines

Hough transform line detection from scratch (numpy only).

Every edge pixel votes for all `(rho, theta)` lines that could pass
through it (`x·cos θ + y·sin θ = ρ`). Peaks in the accumulator are
picked out as detected lines, with non-maximum suppression so nearby
peaks collapse into one line. `line_endpoints` clips a detected line
to the image border.

## Run

```bash
python hough_lines/demo.py
python -m pytest tests/test_hough_lines.py -q
```

## Example

Demo on a synthetic 80×100 edge map (a diagonal line plus a
horizontal line):

```
edge pixels: 120
lines found: 2
  rho=-20.1 theta=-90 deg votes=80 endpoints=(0,20)-(99,20)
  rho=-0.1 theta=-45 deg votes=41 endpoints=(0,0)-(79,79)
```

## API

- `hough_transform(edge_map, theta_step=1.0, rho_step=1.0)` →
  `(accumulator, thetas, rhos)`
- `detect_lines(edge_map, threshold=10, ...)` → list of
  `(rho, theta_degrees, votes)`, highest votes first
- `line_endpoints(rho, theta_degrees, shape)` → `((x1, y1), (x2, y2))`
