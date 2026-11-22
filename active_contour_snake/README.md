# Active Contour (Snake) Segmentation

Active contour segmentation from scratch with numpy only — no OpenCV,
no scipy. A closed curve of points evolves to minimise an energy with
three terms (Kass, Witkin & Terzopoulos, 1988):

* **elasticity** (`alpha`) — squared first differences, keeps points
  evenly spaced along the curve;
* **rigidity** (`beta`) — squared second differences, keeps the curve
  smooth;
* **external energy** (`gamma`) — the negative gradient magnitude of a
  Gaussian-smoothed image, pulling the curve toward edges.

## Minimisation: greedy, not a matrix solve

Each point, in turn, moves to the lowest-energy position inside a 5x5
neighbourhood (Williams & Shah, 1992). Each energy term is normalised
to `[0, 1]` over that neighbourhood before the weighted sum, so the
weights stay comparable even though the raw terms live on very
different scales. The greedy update needs no pentadiagonal solve and
is robust in practice, which is why it is used here instead of the
original Kass et al. matrix formulation.

Two details matter and both are easy to get wrong:

* **Smoothing (`sigma = 1.5`).** With `sigma = 1.0` the gradient
  magnitude 10 px from a step edge is ~1e-22 — below what the search
  can feel — and a snake initialised outside the object never moves.
  `sigma = 1.5` spreads the edge over ~4 px each side, so the
  normalised external term still points downhill toward it.
* **`gamma` default is 1.0, not 0.1.** Because the terms are
  normalised to `[0, 1]`, `gamma` lives on the same scale as `alpha`
  and `beta`; with `gamma = 0.1` the internal terms win and the snake
  stalls. The same normalisation is what lets the snake shrink through
  flat regions: far from the edge the external term still ranks the
  inward candidates first.

Coordinates are `(x, y)` everywhere: `x` is the column index, `y` is
the row index. Contours are `(N, 2)` float arrays. Points are kept
inside the image by clipping candidate positions to the bounds, and
sweeps stop early when no point moves by more than 0.5 px.

One honest caveat: on a pixel-quantised edge the fitted contour
zigzags between adjacent edge pixels, so its measured length
(~121 px in the demo) is larger than `2*pi*r` (~94 px) even though
the mean radius is right. The enclosed area, which is insensitive to
that zigzag, lands within ~2 px^2 of `pi*r^2`.

## Usage

```python
import numpy as np
from active_contour_snake import ActiveContour

image = np.load("image.npy")  # 2D grayscale
snake = ActiveContour(alpha=0.5, beta=0.2, gamma=1.0,
                      n_iterations=100, n_points=50)
initial = snake.initialise_circle(center=(32.0, 32.0), radius=25.0,
                                  image_shape=image.shape)
final = snake.fit(image, initial)
print(snake.energy(image, initial), snake.energy(image, final))
```

`initialise_circle(center, radius, n_points=None, image_shape=None)`
returns an `(N, 2)` array; `fit(image, contour)` returns the evolved
contour without modifying its input; `energy(image, contour)` returns
the total energy (lower is better). `contour_length(contour)` and
`enclosed_area(contour)` (shoelace formula) are module-level helpers.

All inputs are validated: the image must be a non-empty 2D array of
finite values, the contour an `(N, 2)` array with `N >= 3` and finite
values, `radius > 0`, the centre inside the image when `image_shape`
is given, `alpha >= 0`, `beta >= 0`, `gamma > 0`,
`n_iterations >= 1`, `n_points >= 3`. Violations raise `ValueError`.

## Run the demo

```bash
python active_contour_snake/demo.py
```

The demo builds a 64x64 image with a bright disc of radius 15 px at
(32, 32), starts a circle of radius 25 px around the same centre, and
fits the snake.

## Demo output

```
Active contour (snake) segmentation — synthetic disc
Image: 64x64, disc centre (32.0, 32.0), true radius 15.0 px
Initial mean radius: 25.00 px
Final mean radius:   14.99 px
True radius:         15.00 px
Absolute error:      0.01 px
Energy before fit:   247.97
Energy after fit:    147.34
Energy decreased:    True
Final contour length: 120.77 px (2*pi*r = 94.25)
Final enclosed area:  708.78 px^2 (pi*r^2 = 706.86)
```

## Tests

```bash
pytest tests/test_active_contour_snake.py -q
```

21 tests: convergence to the true radius, energy decrease, in-bounds
and finite contour (including on a blank image), determinism, input
contour not modified, circle length/area vs `2*pi*r` / `pi*r^2`,
circle initialisation, and `ValueError` on a 1D image, an empty
image, a NaN image, a contour with < 3 points, a wrong contour shape,
`radius <= 0`, a centre outside the image, negative `alpha`/`beta`,
`gamma <= 0`, and `n_iterations = 0`.
