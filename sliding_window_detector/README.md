# sliding_window_detector

A sliding-window object detector built from scratch with numpy only. The
pipeline is the classic pre-deep-learning recipe:

1. **Sliding window** — a 16×16 window scans the image on a stride of 4.
2. **HOG-like features** — per window, gradient magnitudes are binned into
   9 orientation bins per 4×4 cell, with L2 block normalisation.
3. **Linear classifier** — binary logistic regression trained by gradient
   descent on synthetic circle (positive) vs blank/noise/square (negative)
   windows.
4. **Non-maximum suppression** — overlapping detections are merged by IoU
   so each object is reported once.

## Run

```bash
python sliding_window_detector/demo.py
```

## Demo output

Trains on 60 positive and 60 negative windows, then detects the two
circles in a 48×48 synthetic scene (expected: 2 detections after NMS, 0 on
a blank scene).

## Tests

```bash
python -m pytest sliding_window_detector/tests/test_sliding_window_detector.py -v
```
