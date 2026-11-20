"""Demo: template-matching OCR on rendered text.

Renders "HELLO 123" with the built-in 5x7 bitmap font (scale 2),
recognises it back with template matching, and repeats the test after
flipping a few pixels to show the matcher tolerates light noise.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ocr_template.model import TemplateOCR

TEXT = "HELLO 123"


def main():
    ocr = TemplateOCR()
    image = ocr.render(TEXT, scale=2)
    text, scored = ocr.recognize_with_scores(image)
    print(f"rendered: {TEXT!r} -> image {image.shape[1]}x{image.shape[0]}")
    print(f"recognised: {text!r} (exact match: {text == TEXT})")
    print("per-character Hamming distances (out of 35):",
          [d for _, d in scored])

    rng = np.random.default_rng(4)
    noisy = image.copy()
    flips = rng.choice(image.size, size=12, replace=False)
    noisy.ravel()[flips] = 1.0 - noisy.ravel()[flips]
    noisy_text, noisy_scored = ocr.recognize_with_scores(noisy)
    print(f"with 12 flipped pixels: {noisy_text!r} "
          f"(exact match: {noisy_text == TEXT}), "
          f"max distance {max(d for _, d in noisy_scored)}")
    return text


if __name__ == "__main__":
    main()
