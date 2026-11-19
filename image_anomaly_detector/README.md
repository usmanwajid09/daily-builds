# Image Anomaly Detector (PCA reconstruction error)

Detect anomalous regions in a grayscale image by learning what *normal*
looks like. The image is split into small patches; a PCA model is fitted
on patches from normal images (via `np.linalg.svd`), and each new patch
is scored by its mean squared reconstruction error. A patch the model
cannot reconstruct — a bright block, a scratch, a foreign object — has
a far larger error than a normal patch and is flagged as anomalous when
its error exceeds a threshold (the 95th percentile of the training
errors by default).

## Why

Smooth, low-frequency image content lives close to a low-dimensional
linear subspace, so a few principal components reconstruct it almost
exactly. Anything that breaks that structure sticks out in
reconstruction error, with no labelled anomalies needed for training.

## Usage

```bash
python -m image_anomaly_detector.demo
```

```python
from image_anomaly_detector import (
    PCAAnomalyDetector,
    detect_anomalies,
    extract_patches,
    train_detector,
)

detector = train_detector(normal_images, patch_size=8, n_components=0.95)
mask, errors = detect_anomalies(image, detector, patch_size=8)
print(mask)    # 2D bool array over the patch grid, True = anomalous
print(errors)  # 2D array of per-patch reconstruction errors
```

`n_components` is an int (number of components to keep) or a float in
(0, 1] (fraction of total variance to explain). `detector.set_threshold`
can move the threshold after fitting, either by percentile of the
training errors or by an explicit value.

## Demo

Trains on 20 synthetic normal 32x32 images (smooth linear gradient
plus small Gaussian noise), then scores a fresh normal image and an
anomalous one (same gradient with a bright 8x8 block inserted). Run it
from any directory with `python image_anomaly_detector/demo.py`.

```
trained on 20 normal 32x32 images, 320 patches, 1 components (0.998 variance)
mean train reconstruction error: 0.9636
threshold (2x max train error):  3.2270
normal image:    mean error 1.0120, max error 1.2816, anomalous patches flagged: 0
anomalous image: mean error 3.0891, max error 34.0612, anomalous patches flagged: 1
anomaly flagged: True, normal clean: True
```

The anomalous patch's reconstruction error (34.06) is ~10x the
threshold, while the worst normal patch (1.28) stays well below it.

## Run tests

```bash
python -m pytest tests/test_image_anomaly_detector.py -v
```

13 tests: patch extraction shapes/counts, low-rank reconstruction
round-trip, training patches below threshold, bright-block error far
above normal error, `predict` flagging, invalid-input `ValueError`s,
threshold set after fit, and explained-variance sums.
