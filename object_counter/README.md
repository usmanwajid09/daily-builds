# object_counter

Count objects in a binary image from scratch — connected-components
labelling with a hand-written two-pass algorithm and union-find.
NumPy only: no OpenCV, no SciPy.

## What it does

- `label_image(binary_image, connectivity=8)` — labels each connected
  foreground component with a unique integer (background is 0).
- `count_objects(binary_image, min_size=1, connectivity=8)` — returns
  one entry per object with:
  - `label` — component number (scan order)
  - `area` — pixel count
  - `centroid` — `(row, col)` as floats
  - `bbox` — `(min_row, min_col, max_row, max_col)`, inclusive
- `min_size` drops small objects (e.g. speckle noise); `connectivity`
  is 4 (edge neighbours only) or 8 (diagonals count too).

Input must be a 2D NumPy array of 0/1 integers or booleans; bad
shapes, values, `min_size < 1` or other connectivity raise `ValueError`
with a clear message.

## How to run

From the repo root:

```bash
/home/hatch/workspace/db-venv/bin/python object_counter/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_object_counter.py -q
```

## Demo output

```
Image: 40x50, foreground pixels: 81
Raw components (8-connectivity, no filtering): 17
Objects with area >= 4: 5
  label 1: area=24 centroid=(4.5, 4.5) bbox=(2, 3, 7, 6)
  label 5: area=25 centroid=(14.0, 22.0) bbox=(12, 20, 16, 24)
  label 9: area= 6 centroid=(22.5, 32.5) bbox=(20, 30, 25, 35)
  label 13: area= 9 centroid=(29.0, 41.0) bbox=(28, 40, 30, 42)
  label 16: area= 5 centroid=(35.0, 7.0) bbox=(33, 5, 37, 9)
Objects with area >= 4 (4-connectivity): 3
Objects with area >= 1: 17 (includes 12 noise specks)
```

Labels keep their original component numbers after `min_size`
filtering (labels 2–4, 6–8, … are the filtered-out noise specks).

The demo image (seed 42) holds three solid blobs, two diagonal chains
that only touch diagonally, and 12 isolated noise pixels. With
`min_size=4` the noise is filtered out; with 4-connectivity the two
diagonal chains split apart, leaving only the 3 solid blobs.

## Tests

16 pytest tests in `tests/test_object_counter.py`: correctness
(centroid/bbox/area, merging, connectivity, random-image check against
an independent flood fill), edge cases (empty image, single pixel,
full image, boolean input) and input validation (bad shape, bad
values, bad `min_size`, bad connectivity, non-array, empty array).
