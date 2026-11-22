"""Demo: simplified SIFT features on a synthetic image.

Builds a 128x128 grayscale image — a bright square, a bright disc and a
smooth Gaussian blob on a black background — detects SIFT-like keypoints,
prints how many were found and the descriptor shape/norm, then shifts the
image by (5, 7) pixels and matches descriptors between the two versions.
"""

import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sift_features.sift_features import SIFTFeatures, match_descriptors


def make_test_image(height=128, width=128):
    """Synthetic image: bright square + disc + Gaussian blob on black."""
    image = np.zeros((height, width), dtype=np.float64)
    # Bright square, top-left.
    image[20:45, 20:45] = 1.0
    # Bright disc, bottom-right.
    rows, cols = np.mgrid[0:height, 0:width]
    image[(rows - 95) ** 2 + (cols - 95) ** 2 <= 12 ** 2] = 1.0
    # Smooth Gaussian blob, bottom-left.
    r2 = (rows - 95) ** 2 + (cols - 30) ** 2
    image += 0.9 * np.exp(-r2 / (2 * 6.0 ** 2))
    return np.clip(image, 0.0, 1.0)


def main():
    image = make_test_image()
    sift = SIFTFeatures()
    keypoints, descriptors = sift.detect_and_describe(image)
    print(f"Image shape: {image.shape}")
    print(f"Keypoints detected: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape}")
    if len(descriptors):
        norms = np.linalg.norm(descriptors, axis=1)
        print(f"Descriptor norms: min={norms.min():.4f} "
              f"max={norms.max():.4f} mean={norms.mean():.4f}")
        print("First 3 keypoints (y, x, scale, orientation):")
        for kp in keypoints[:3]:
            print(f"  y={kp['y']:.1f} x={kp['x']:.1f} "
                  f"scale={kp['scale']:.2f} orientation={kp['orientation']:.3f}")

    # Shifted copy: same content moved by (dy=5, dx=7).
    shifted = np.roll(np.roll(image, 5, axis=0), 7, axis=1)
    keypoints2, descriptors2 = sift.detect_and_describe(shifted)
    matches = match_descriptors(descriptors, descriptors2, ratio=0.75)
    print(f"Keypoints in shifted image: {len(keypoints2)}")
    print(f"Good matches (ratio test 0.75): {len(matches)}")
    if matches:
        dists = [m[2] for m in matches]
        print(f"Match distances: min={min(dists):.4f} "
              f"max={max(dists):.4f} mean={np.mean(dists):.4f}")
        print("First 3 matches (i, j, distance) with keypoint positions:")
        for i, j, dist in matches[:3]:
            k1, k2 = keypoints[i], keypoints2[j]
            print(f"  ({k1['y']:.1f}, {k1['x']:.1f}) -> "
                  f"({k2['y']:.1f}, {k2['x']:.1f})  d={dist:.4f}")


if __name__ == "__main__":
    main()
