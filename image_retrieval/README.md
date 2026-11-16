# Image Retrieval (colour-histogram CBIR)

Content-based image retrieval by example: images are indexed by their
normalised RGB colour histogram (8 bins per channel, 512 bins), and a
query image is ranked against the index by Pearson correlation between
histograms. Images with similar colour content come back first.

## Why

Colour histograms are the classic CBIR baseline — simple, fast, and
surprisingly effective for finding visually similar images when colour
is the main signal. This is the starting point before adding texture or
shape features to the ranking.

## Usage

```bash
pip install -r requirements.txt
python -m image_retrieval.demo
```

The demo indexes 20 synthetic images in 5 colour themes (sunset,
forest, ocean, desert, night), queries with a fresh "ocean" image, and
prints the top-5 ranked results with scores, plus a self-query check
that must score 1.0.

```python
from image_retrieval import ImageIndex, make_gradient

index = ImageIndex()
index.add("sunset-0", make_gradient((255, 120, 40), (255, 180, 90)))
print(index.query(make_gradient((255, 120, 40), (255, 180, 90)), top_k=3))
```

## Run tests

```bash
python -m pytest tests/test_image_retrieval.py -v
```
