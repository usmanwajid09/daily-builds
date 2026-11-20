# Histogram Matching

Histogram specification: remap a source image so its histogram
matches a reference image's histogram (numpy only).

The cumulative histograms of source and reference are computed and a
lookup table maps each source level to the smallest reference level
with an equal-or-greater CDF value. Grayscale images are matched as
one channel; RGB images are matched per channel.

## Run

```bash
python histogram_matching/demo.py
python -m pytest tests/test_histogram_matching.py -q
```

## Example

```python
from histogram_matching import match_histogram

matched = match_histogram(source, reference)  # same dtype/range as source
```

Demo on synthetic uint8 images (dark low-contrast source, bright
high-contrast reference):

```
source:    mean= 59.26 std= 16.92 range=[13, 112]
reference: mean=160.19 std= 48.87 range=[35, 255]
matched:   mean=161.67 std= 48.92 range=[35, 255]
```

The matched image keeps the source's content but takes on the
reference's brightness and contrast almost exactly (mean within 1.5
grey levels, std within 0.1).

## API

- `match_histogram(source, reference, bins=256)` → image in the source's dtype and range
- `compute_histogram(image, bins=256)` → integer counts, length `bins`
- `compute_cdf(image, bins=256)` → normalised cumulative histogram
