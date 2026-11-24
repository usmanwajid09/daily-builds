"""Demo: licence-plate detection on a synthetic scene.

Builds a 240x320 grayscale scene with three known plates, runs
:func:`plate_detector.plate.detect_plates` on it, and prints the
detected count, bounding boxes, aspect ratios and crop sizes, plus
how the detections match the ground truth (IoU) and the false-positive
count on a blank scene.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python plate_detector/demo.py`):
# put the repo root on sys.path so the package import below resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plate_detector.plate import (
    detect_plates,
    extract_plate_crop,
    make_plate_scene,
    match_detections,
)

# (x, y, width, height) of the three planted plates.
PLATES = [(30, 40, 72, 18), (180, 90, 60, 16), (90, 170, 80, 20)]


def main():
    image, truth = make_plate_scene(240, 320, plate_specs=PLATES, seed=7)
    detections = detect_plates(image)
    n_matched, pairs = match_detections(detections, truth)

    print(f"scene: {image.shape[1]}x{image.shape[0]} px, "
          f"planted plates: {len(truth)}")
    print(f"detected plates: {len(detections)} "
          f"(matched vs ground truth: {n_matched}/{len(truth)})")
    for i, det in enumerate(detections, 1):
        crop = extract_plate_crop(image, det)
        iou = next((p[2] for p in pairs if p[0] is det), 0.0)
        print(f"  plate {i}: box=({det['x']}, {det['y']}, "
              f"{det['width']}, {det['height']}) "
              f"aspect={det['aspect_ratio']:.2f} area={det['area']} px "
              f"crop={crop.shape[1]}x{crop.shape[0]} px IoU={iou:.3f}")

    blank, _ = make_plate_scene(240, 320, plate_specs=[], seed=7)
    blank_detections = detect_plates(blank)
    print(f"blank scene: detected {len(blank_detections)} plates "
          f"(false positives: {len(blank_detections)})")

    # A second, busier scene with randomly placed plates.
    busy, busy_truth = make_plate_scene(240, 320, n_plates=5, seed=42)
    busy_detections = detect_plates(busy)
    busy_matched, _ = match_detections(busy_detections, busy_truth)
    print(f"random scene: planted {len(busy_truth)}, detected "
          f"{len(busy_detections)}, matched {busy_matched}")


if __name__ == "__main__":
    main()
