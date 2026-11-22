# Horn-Schunck Optical Flow

Dense optical flow from scratch (numpy only), using the classic
Horn & Schunck (1981) algorithm.

Given two grayscale frames, it estimates a flow field `(u, v)` — the
horizontal and vertical displacement of every pixel — by minimising
the brightness-constancy error plus a smoothness penalty on the flow
field. The estimate is found with the iterative update

```
u = ū − Ix·(Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)
v = v̄ − Iy·(Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)
```

where `ū, v̄` are 4-neighbour averages and `Ix, Iy, It` are the
spatial/temporal derivatives from the standard Horn-Schunck 2×2
kernels, averaged over the two frames. The iteration is fully
vectorised numpy (loops over iterations only, never over pixels).

## Run

```bash
python horn_schunck_flow/demo.py
python -m pytest tests/test_horn_schunck_flow.py -q
```

## Example

Demo on a 64×64 image: a bright Gaussian blob (σ = 5) on a dark
background, translated by (dx, dy) = (+3, +2) px in the second frame,
with α = 1.0 and 100 iterations:

```
image: 64x64 grayscale, blob shifted by (dx, dy) = (3, 2)
iterations run: 100
true shift: dx = 3.00 px, dy = 2.00 px
recovered mean flow in object region: u = 2.662 px, v = 1.772 px
recovery error: du = 0.338 px, dv = 0.228 px
mean flow magnitude in static background: 0.0050 px
overall mean flow: u = 0.698 px, v = 0.465 px
overall mean flow magnitude: 0.933 px
```

The flow recovers the known translation to within ~0.34 px inside
the moving blob, while the static corner background stays essentially
still (0.005 px).

## API

- `HornSchunckFlow(alpha=1.0, n_iterations=100, tolerance=1e-4)`
  - `compute_flow(img1, img2)` → `(u, v)` float64 arrays, same shape
    as the inputs; `u` is horizontal (positive = right), `v` is
    vertical (positive = down). Stops early when the largest per-step
    change falls below `tolerance`; `iterations_run_` records how many
    iterations actually ran.
- `flow_magnitude(u, v)` → per-pixel speed `sqrt(u² + v²)`
- `flow_angle(u, v)` → per-pixel direction in degrees, in (−180, 180];
  0° is right, 90° is down (image coordinates)
- `mean_flow(u, v)` → `(mean_u, mean_v)` over the whole field
- `warp_image(image, u, v)` → frame 2 warped by the flow, which
  approximately reconstructs frame 1 (bilinear, numpy only)

Inputs are validated everywhere: images must be 2D grayscale arrays
of the same shape, non-empty, with finite values; `alpha` must be
positive, `n_iterations >= 1`, `tolerance >= 0` — violations raise
`ValueError`.
