# plate_detector

Licence-plate detection in synthetic grayscale images, built from
scratch with **numpy + stdlib only** (no OpenCV, no scipy).

## How it works

1. **Synthetic scenes** (`make_plate_scene`): a dark background
   (~40 grey + Gaussian noise + a few bright specks) with bright
   rectangular plates (~220 grey, width ~3–5× height, e.g. 60×16 px)
   holding dark character blocks. Scenes are deterministic for a
   given `seed`; plates can be placed explicitly or at random.
2. **Threshold** (`threshold_image`): pixels brighter than 150
   become the foreground mask.
3. **Morphology** (`binary_closing`): a 3×3 closing (dilation then
   erosion, implemented with numpy shifts) bridges the small dark
   gaps between a plate's characters so each plate is one solid
   component. Specks in the scene are kept ≥ 4 px from plates, since
   closing bridges gaps of ≤ 2 px and a closer speck would fuse with
   a plate and stretch its box.
4. **Connected components** (`label_components`): 4-connected flood
   fill, then a bounding box + bright-pixel area per component.
5. **Filtering** (`detect_plates`): a component counts as a plate
   when its aspect ratio is in [2.0, 6.0] and its area is ≥ 300 px.
   This rejects square blobs, thin noise, and the isolated specks.
6. **Crop extraction** (`extract_plate_crop`): returns the cropped
   sub-image for a detected box (dict or `(x, y, w, h)` tuple).

`box_iou` / `match_detections` score detections against ground truth
for evaluation.

## Usage

```python
from plate_detector.plate import (
    detect_plates, extract_plate_crop, make_plate_scene, match_detections,
)

image, truth = make_plate_scene(240, 320, n_plates=3, seed=7)
plates = detect_plates(image)
print(len(plates), "plates found")
for plate in plates:
    crop = extract_plate_crop(image, plate)
    print(plate, "crop:", crop.shape)

n_matched, pairs = match_detections(plates, truth)
print(f"matched {n_matched}/{len(truth)}")
```

## Demo

Run from the repo root:

```bash
python plate_detector/demo.py
```

Output:

```text
scene: 320x240 px, planted plates: 3
detected plates: 3 (matched vs ground truth: 3/3)
  plate 1: box=(30, 40, 72, 18) aspect=4.00 area=936 px crop=72x18 px IoU=1.000
  plate 2: box=(180, 90, 60, 16) aspect=3.75 area=710 px crop=60x16 px IoU=1.000
  plate 3: box=(90, 170, 80, 20) aspect=4.00 area=1096 px crop=80x20 px IoU=1.000
blank scene: detected 0 plates (false positives: 0)
random scene: planted 5, detected 5, matched 5
```

A sweep over 30 random 5-plate scenes detects 5/5 with IoU ≥ 0.99
every time, and 10 blank scenes produce 0 false positives.

## Tests

26 pytest tests in `tests/test_plate_detector.py`, covering detection
(single / multiple / random scenes), exact bounding boxes, crop
extraction, morphology (closing bridges character gaps), rejection
of square blobs / tiny strips / blank and noise-only scenes, IoU and
matching, determinism, and invalid input (non-2D / empty images, bad
threshold, bad filter params, out-of-bounds or malformed crop boxes,
bad kernel sizes).

```bash
python -m pytest tests/test_plate_detector.py -q
# 26 passed
```
