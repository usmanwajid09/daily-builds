# HOG Features

Histogram of Oriented Gradients (HOG) descriptor, from scratch. The
image is split into cells; each cell votes into an orientation
histogram weighted by gradient magnitude. Cells are grouped into
overlapping blocks, each block is L2-normalised with L2-Hys clipping,
and the blocks are concatenated into one feature vector.

## Why

HOG is the classic hand-crafted feature for object detection (Dalal &
Triggs, 2005) — it powered pedestrian detection long before deep
learning. Building it by hand shows exactly what a gradient histogram
captures: edge directions, not raw pixel values.

## Usage

```bash
pip install -r requirements.txt
python hog_features/demo.py
```

This computes HOG descriptors for a vertical bar and a horizontal bar
and compares them.

Example output:

```
Feature vector length: 324 (expected 324)
Vertical bar descriptor norm: 3.000
Horizontal bar descriptor norm: 3.000
Euclidean distance between the two descriptors: 4.243
Descriptors differ: True (vertical edges vs horizontal edges)
Blank image descriptor norm: 0.000 (no gradients, all zeros)
```

## Run tests

```bash
python -m pytest tests/test_hog_features.py -q
```
