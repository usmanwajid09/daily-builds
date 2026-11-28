# Face Detector (Simple)

Simple face detection via skin-colour segmentation from scratch
(numpy only).

Pixels whose YCbCr chroma falls in the skin range (Cb 77–127,
Cr 133–173) are segmented, connected skin regions are labelled, and
regions that are face-sized, roughly elliptical and not too stretched
are reported as faces. A face ellipse fills about 0.79 of its bounding
box, while a solid rectangle fills 1.0 and scattered skin fills far
less — that fill-ratio check is what separates a face from other
skin-coloured blobs. No trained model is involved, so it works best
on clear, well-lit synthetic or studio-style images.

## Run

```bash
python face_detector_simple/demo.py
python -m pytest tests/test_face_detector_simple.py -q
```

## Example output

Demo on an 80×100 synthetic scene — a skin-coloured face ellipse on
a blue background, plus a small skin square and a large skin
rectangle as distractors:

```
image: 100x80 RGB
skin pixels: 1650
faces detected: 1
  face box (top, left, bottom, right): (18, 34, 63, 67)
true face box: (18, 34, 63, 67)
```

Exactly one face is found, and its bounding box matches the planted
ellipse exactly; both distractors are rejected (too small, and not
elliptical).

## API

- `detect_faces(image, min_area=200, max_fill=0.95, min_fill=0.45, min_aspect=0.5, max_aspect=2.0)` → list of `(top, left, bottom, right)` boxes, `bottom`/`right` exclusive
- `skin_mask(image, cb_range=(77, 127), cr_range=(133, 173))` → boolean skin mask
- `rgb_to_ycbcr(image)` → YCbCr image (ITU-R BT.601)
