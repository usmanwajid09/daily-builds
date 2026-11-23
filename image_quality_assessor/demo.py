"""Demo: assess synthetic images of degrading quality.

Builds a sharp checkerboard, then a blurred, a noisy, a low-contrast
and a dark version of it, and prints each one's score and verdict.
Quality should fall monotonically as the degradation gets worse:
sharp scores highest, the degraded versions lower.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python image_quality_assessor/demo.py`):
# make the repo root importable so `image_quality_assessor` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_quality_assessor.quality import (
    add_gaussian_noise,
    assess,
    box_blur,
    make_checkerboard,
    make_low_contrast,
)


def main():
    sharp = make_checkerboard(size=64, square=8)
    variants = [
        ("sharp", sharp),
        ("blurred (box blur x12)", box_blur(sharp, iterations=12)),
        ("noisy (sigma=30)", add_gaussian_noise(sharp, sigma=30.0, seed=7)),
        ("low contrast", make_low_contrast(sharp)),
        ("dark (x0.12)", sharp * 0.12),
    ]
    results = []
    for name, image in variants:
        result = assess(image)
        results.append((name, result))
        print(
            f"{name:24s} score={result['score']:5.1f} "
            f"verdict={result['verdict']:<9s} "
            f"blur={result['blur']:9.1f} noise={result['noise']:5.1f} "
            f"contrast={result['contrast']:5.1f} "
            f"brightness={result['brightness']:5.1f}"
        )
    sharp_score = results[0][1]["score"]
    degraded = [score for _, r in results[1:] for score in [r["score"]]]
    ordering_holds = all(sharp_score > s for s in degraded)
    print(f"\nsharp scores highest: {ordering_holds} "
          f"(sharp {sharp_score:.1f} vs best degraded {max(degraded):.1f})")
    return results


if __name__ == "__main__":
    main()
