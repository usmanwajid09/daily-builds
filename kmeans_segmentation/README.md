# K-Means Segmentation

Image segmentation from scratch: k-means clustering (k-means++
initialisation, Lloyd iterations) groups pixels by colour. The result
is a label map — which cluster each pixel belongs to — and a segmented
image where every pixel is replaced by its cluster centre colour.

## Why

Colour-based segmentation is the simplest way to split an image into
regions, and k-means on pixels is the textbook starting point. It also
shows k-means working on real 2D data instead of toy points.

## Usage

```bash
pip install -r requirements.txt
python kmeans_segmentation/demo.py
```

This builds a synthetic 30x30 image with three colour blocks (red,
green, blue) plus noise, and segments it into 3 clusters.

Example output:

```
Image shape: (30, 30, 3), clusters: 3
Cluster sizes (pixels): [225, 225, 450] (expected ~225, 225, 450)
Cluster centres (RGB):
[[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
Unique colours in segmented image: 3
Labels shape: (30, 30), unique labels: [0, 1, 2]
```

## Run tests

```bash
python -m pytest tests/test_kmeans_segmentation.py -q
```
