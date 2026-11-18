"""Demo: ORB-style features on a shifted synthetic image.

Builds an image with a white square and triangle, shifts it by a known
(5, 8) pixels, extracts FAST keypoints + BRIEF-like descriptors from
both, and brute-force matches them with a Hamming-distance ratio test.
The recovered shift is reported from the matched keypoints.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from orb_features.features import extract_orb, match_descriptors


def make_image(rows=120, cols=160, seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros((rows, cols), dtype=np.float64)
    img[20:50, 20:50] = 255.0  # white square
    for r in range(20, 60):  # white right triangle
        img[r, 90:90 + (r - 19)] = 255.0
    img += rng.normal(0, 3.0, size=img.shape)
    return np.clip(img, 0, 255)


def shift_image(img, dr, dc):
    out = np.zeros_like(img)
    r0, r1 = max(0, dr), min(img.shape[0], img.shape[0] + dr)
    c0, c1 = max(0, dc), min(img.shape[1], img.shape[1] + dc)
    s0, s1 = r0 - dr, r1 - dr
    t0, t1 = c0 - dc, c1 - dc
    out[r0:r1, c0:c1] = img[s0:s1, t0:t1]
    return out


def main():
    img1 = make_image()
    img2 = shift_image(img1, 5, 8)
    kp1, desc1 = extract_orb(img1, n_contiguous=9)
    kp2, desc2 = extract_orb(img2, n_contiguous=9)
    print(f"Image 1: {len(kp1)} keypoints, descriptors {desc1.shape}")
    print(f"Image 2: {len(kp2)} keypoints, descriptors {desc2.shape}")
    matches = match_descriptors(desc1, desc2)
    print(f"Matches after ratio test: {len(matches)}")
    if matches:
        shifts = []
        for i, j, dist in matches:
            r1, c1 = kp1[i]
            r2, c2 = kp2[j]
            shifts.append((r2 - r1, c2 - c1))
        shifts = np.array(shifts)
        median_shift = np.median(shifts, axis=0)
        print(f"Median recovered shift (dr, dc): {median_shift.tolist()} "
              f"(true shift: [5, 8])")
        correct = np.sum(np.all(shifts == [5, 8], axis=1))
        print(f"Matches with exactly the true shift: {correct}/{len(matches)}")
        print("Best 5 matches (i, j, hamming distance):")
        for m in matches[:5]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
