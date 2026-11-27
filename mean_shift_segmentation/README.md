# Mean-Shift Segmentation

Segments an image with the mean-shift algorithm: pixels are mapped into
a joint colour + spatial feature space and iteratively shifted towards
the local mode of a Gaussian kernel density estimate. Pixels whose
modes end up close together form one segment.

## How it works

1. Each pixel becomes a 5-D feature vector: RGB channels scaled to
   [0, 1] plus row/column coordinates scaled by `spatial_weight`.
2. Mean shift: each point is repeatedly replaced by the Gaussian
   kernel-weighted mean of all points, until it stops moving (`tol`).
3. Modes closer than `bandwidth` are merged into segment centres.
4. Every pixel is assigned to its nearest centre; the label map is
   returned as an int32 array of shape (H, W).

`segment_image` returns a posterized RGB image where each pixel is
replaced by the mean colour of its segment.

## Run

```bash
python mean_shift_segmentation/demo.py
python -m pytest tests/test_mean_shift_segmentation.py -q
```

## Example output

```
Scene shape: (32, 40, 3), dtype: uint8
Segments found: 3
Segment sizes (pixels): [1012, 148, 120]
Segmented image shape: (32, 40, 3), dtype: uint8
Distinct colours in segmented image: 3
```
