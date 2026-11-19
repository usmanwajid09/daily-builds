# skeletonization

Skeletonization by Zhang-Suen thinning (numpy only).

Thinning peels boundary pixels off a binary shape until only a
1-pixel-wide skeleton remains — the shape's centreline — keeping
the original topology (a ring stays a ring, a bar stays one line).
The skeleton makes shape features easy to count: endpoints
(skeleton pixels with exactly one neighbour) mark stroke ends, and
branch points (three or more neighbours) mark junctions. Neighbour
counts use 8-connectivity, so a junction in a plus sign counts as a
small cluster: the centre pixel and its four diagonal neighbours
each see 3+ neighbours.

## API

- `zhang_suen_thin(binary, max_iterations=1000)` — returns
  `(skeleton, iterations)`; the skeleton is a 2-D boolean array.
- `count_endpoints(skeleton)` — pixels with exactly one neighbour.
- `count_branch_points(skeleton)` — pixels with 3+ neighbours.
- `neighbour_count(binary)` — foreground-neighbour count per pixel.

## How to run

```bash
python skeletonization/demo.py
python -m pytest tests/test_skeletonization.py -q
```

## Demo output

```text
plus sign: 225 foreground pixels -> 39 skeleton pixels (4 iterations)
  endpoints=4, branch points=5
bar: 147 foreground pixels -> 14 skeleton pixels (4 iterations)
  endpoints=2, branch points=0
```

18 tests pass.
