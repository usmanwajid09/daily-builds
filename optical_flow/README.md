# optical_flow

Lucas-Kanade optical flow from scratch (numpy only).

For a small window around a point, brightness constancy gives
`Ix * u + Iy * v + It = 0` at every pixel; the flow `(u, v)` is the
least-squares solution over the window. Windows whose gradient matrix
is (near-)singular — flat regions, or a single straight edge (the
aperture problem) — are reported as zero flow.

## API

- `spatial_gradients(image)` — returns `(Ix, Iy)` via central
  differences (`numpy.gradient`).
- `lucas_kanade(prev, curr, points, window_size=15, min_eigenvalue=1e-3)`
  — returns an `(N, 2)` float array of `(u, v)` displacements for the
  given `(x, y)` points. Unreliable points get `(0, 0)`.
- `dense_flow(prev, curr, window_size=15, step=8, min_eigenvalue=1e-3)`
  — flow on a regular grid; returns `(U, V, xs, ys)`.

## How to run

```bash
python optical_flow/demo.py
python -m pytest tests/test_optical_flow.py -q
```

## Demo output

```
true shift: (dx=3, dy=2)
flow at blob centre (40, 30): u=2.778, v=1.857
dense flow grid: 5x7 points
mean flow magnitude over grid: 3.430
max flow magnitude over grid: 7.225
flow at grid centre (37, 27): u=2.845, v=2.075
```

The demo shifts a Gaussian blob by (dx=3, dy=2) between two 60x80
frames; Lucas-Kanade recovers (2.78, 1.86) at the blob centre. The
dense-grid mean magnitude is inflated by edge windows away from the
blob, where the aperture-limited estimates overshoot. 10 tests pass.
