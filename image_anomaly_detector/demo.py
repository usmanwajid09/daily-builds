"""Demo: PCA reconstruction-error anomaly detection on synthetic images.

Trains a :class:`PCAAnomalyDetector` on 20 normal 32x32 images — each a
smooth linear gradient plus small Gaussian noise — then scores a fresh
normal image and an anomalous one (the same gradient with a bright
8x8 square block inserted). The normal image should flag no patches;
the anomalous image should flag at least the patch holding the block.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python image_anomaly_detector/demo.py`):
# make the repo root importable so `image_anomaly_detector` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_anomaly_detector.anomaly import (
    detect_anomalies,
    extract_patches,
    train_detector,
)

SIZE = 32
PATCH_SIZE = 8
BLOCK = (8, 8, 16, 16)  # row_min, col_min, row_max, col_max (exclusive)


def make_normal_image(seed):
    """Return a 32x32 smooth gradient image with small noise."""
    rng = np.random.default_rng(seed)
    rows, cols = np.indices((SIZE, SIZE))
    gradient = 40.0 + 2.0 * cols + 1.5 * rows
    image = gradient + rng.normal(0.0, 1.0, size=(SIZE, SIZE))
    return np.clip(image, 0.0, 255.0)


def make_anomalous_image(seed):
    """Return the normal image with a bright 8x8 block inserted."""
    image = make_normal_image(seed)
    r_min, c_min, r_max, c_max = BLOCK
    image[r_min:r_max, c_min:c_max] = 255.0
    return image


def main():
    normal_images = [make_normal_image(seed) for seed in range(20)]
    detector = train_detector(
        normal_images, patch_size=PATCH_SIZE, n_components=0.95
    )
    # Mean training error, recomputed from the training patches.
    train_patches = np.concatenate(
        [extract_patches(img, PATCH_SIZE) for img in normal_images],
        axis=0,
    )
    train_errors = detector.reconstruction_error(train_patches)
    # A fresh normal image should flag nothing, so raise the threshold
    # above the largest training error instead of the 95th percentile.
    detector.set_threshold(value=float(train_errors.max()) * 2.0)

    normal_test = make_normal_image(123)
    anomalous_test = make_anomalous_image(123)
    normal_mask, normal_errors = detect_anomalies(
        normal_test, detector, patch_size=PATCH_SIZE
    )
    anomaly_mask, anomaly_errors = detect_anomalies(
        anomalous_test, detector, patch_size=PATCH_SIZE
    )

    print(f"trained on 20 normal 32x32 images, "
          f"{train_errors.size} patches, "
          f"{detector.n_components_} components "
          f"({detector.explained_variance_ratio_.sum():.3f} variance)")
    print(f"mean train reconstruction error: {train_errors.mean():.4f}")
    print(f"threshold (2x max train error):  {detector.threshold_:.4f}")
    print(f"normal image:    mean error {normal_errors.mean():.4f}, "
          f"max error {normal_errors.max():.4f}, "
          f"anomalous patches flagged: {int(normal_mask.sum())}")
    print(f"anomalous image: mean error {anomaly_errors.mean():.4f}, "
          f"max error {anomaly_errors.max():.4f}, "
          f"anomalous patches flagged: {int(anomaly_mask.sum())}")
    print(f"anomaly flagged: {bool(anomaly_mask.any())}, "
          f"normal clean: {not bool(normal_mask.any())}")
    return {
        "mean_train_error": float(train_errors.mean()),
        "threshold": float(detector.threshold_),
        "normal_flagged": int(normal_mask.sum()),
        "anomalous_flagged": int(anomaly_mask.sum()),
    }


if __name__ == "__main__":
    main()
