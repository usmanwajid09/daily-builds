# Watershed Segmentation

Simple watershed segmentation from scratch (numpy only).

The distance transform of a binary mask is computed with a two-pass
chamfer algorithm, its regional maxima become markers, and a priority
flood (highest distance first) grows each marker's basin until basins
meet, where a watershed line (label `-1`) is drawn. Touching objects
that are separate in the distance transform — e.g. two overlapping
discs — are split into two segments.

## Run

```bash
python watershed_segmentation/demo.py
python -m pytest tests/test_watershed_segmentation.py -q
```

## Example

Demo on a 60×80 mask of two overlapping discs (radius 14):

```
mask: 60x80, foreground pixels: 1119
segments found: 2
watershed line pixels: 38
  segment 1: 551 pixels
  segment 2: 530 pixels
```

## API

- `distance_transform(mask)` → float array of distances to background
- `find_markers(mask, min_distance=1.0)` → int array of marker regions
- `watershed(mask, markers=None, min_distance=5.0)` → int label map
  (positive ids = segments, `0` = background, `-1` = watershed line)
- `segment_count(labels)` → number of segments
- `watershed_lines(labels)` → bool mask of watershed-line pixels
