"""Demo: crop-health detection on a synthetic multispectral field.

Run from the repo root:
    python crop_health_detector/demo.py
"""

import os
import sys

# Allow running as `python crop_health_detector/demo.py` from the repo root:
# the script's own directory is on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from crop_health_detector.crop_health import (
    LABEL_NAMES,
    classification_accuracy,
    evi,
    health_map,
    make_field,
    ndvi,
    zone_mean_ndvi,
)


def main():
    height, width, seed = 64, 64, 42
    red, nir, truth = make_field(height, width, seed=seed)
    ndvi_map = ndvi(red, nir)
    evi_map = evi(nir, red)
    labels, summary = health_map(ndvi_map, evi_map)
    accuracy = classification_accuracy(labels, truth)
    means = zone_mean_ndvi(ndvi_map, truth)

    print("Crop Health Detector — synthetic multispectral field")
    print(f"Field: {height}x{width} px (seed={seed}), bands: red + NIR")
    print(f"NDVI range: [{ndvi_map.min():.3f}, {ndvi_map.max():.3f}]  "
          f"mean: {ndvi_map.mean():.3f}")
    print(f"EVI2 range: [{evi_map.min():.3f}, {evi_map.max():.3f}]  "
          f"mean: {evi_map.mean():.3f}")
    print()
    print("Mean NDVI per ground-truth zone:")
    for name in LABEL_NAMES:
        print(f"  {name:<9}: {means[name]:.3f}")
    print()
    print("Health map (share of field):")
    for name in LABEL_NAMES:
        print(f"  {name:<9}: {summary['counts'][name]:>5} px  "
              f"({summary['percentages'][name]:5.1f}%)")
    print()
    print(f"Classification accuracy vs ground truth: {accuracy * 100:.1f}%")
    print(f"Healthy area: {summary['percentages']['healthy']:.1f}%  |  "
          f"Stressed area: {summary['percentages']['stressed']:.1f}%  |  "
          f"Bare soil: {summary['percentages']['bare']:.1f}%")
    # Sanity check on the planted zones: the two healthy quarters should
    # dominate the map and accuracy should clear 90%.
    assert accuracy >= 0.90, f"accuracy {accuracy:.3f} below 0.90 target"
    assert means["healthy"] > means["stressed"] > means["bare"]
    print("OK: accuracy >= 90% and healthy > stressed > bare NDVI ordering holds.")


if __name__ == "__main__":
    main()
