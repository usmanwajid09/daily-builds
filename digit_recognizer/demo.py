"""Demo: recognise noisy 8x8 digit patterns by template matching.

Generates 20 noisy variants (5% pixel flips) of each digit 0-9, runs
:func:`digit_recognizer.recognize` on them, and prints overall accuracy
plus per-digit accuracy. A few example predictions are shown with their
winning distance and the runner-up distance, so the margin is visible.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python digit_recognizer/demo.py`):
# make the repo root importable so `digit_recognizer` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digit_recognizer.digits import accuracy, add_noise, recognize, template

N_VARIANTS = 20
FLIP_PROB = 0.05
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    images, labels = [], []
    for digit in range(10):
        clean = template(digit)
        for _ in range(N_VARIANTS):
            images.append(add_noise(clean, FLIP_PROB, rng))
            labels.append(digit)

    overall = accuracy(images, labels)
    print(f"noisy variants: {len(images)} "
          f"({N_VARIANTS} per digit, flip_prob={FLIP_PROB})")
    print(f"overall accuracy: {overall:.3f} "
          f"({int(overall * len(images))}/{len(images)} correct)")

    for digit in range(10):
        subset = [img for img, lab in zip(images, labels) if lab == digit]
        acc = accuracy(subset, [digit] * len(subset))
        print(f"  digit {digit}: accuracy {acc:.2f}")

    print("example predictions (noisy '3', '7', '9'):")
    for digit in (3, 7, 9):
        noisy = add_noise(template(digit), FLIP_PROB, rng)
        pred, scores = recognize(noisy)
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        print(f"  true={digit} predicted={pred} "
              f"best={ranked[0][1]:.3f} runner-up={ranked[1][1]:.3f}")
    return overall


if __name__ == "__main__":
    main()
