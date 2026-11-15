"""Demo: HOG descriptors of a vertical bar vs a horizontal bar."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from hog_features.hog import HOGDescriptor


def vertical_bar(size=32):
    img = np.zeros((size, size))
    img[:, size // 2 - 2:size // 2 + 2] = 1.0
    return img


def horizontal_bar(size=32):
    img = np.zeros((size, size))
    img[size // 2 - 2:size // 2 + 2, :] = 1.0
    return img


def main():
    hog = HOGDescriptor(cell_size=8, block_size=2, n_bins=9)
    v = hog.compute(vertical_bar())
    h = hog.compute(horizontal_bar())
    dist = float(np.linalg.norm(v - h))
    print(f"Feature vector length: {len(v)} (expected {hog.feature_length((32, 32))})")
    print(f"Vertical bar descriptor norm: {np.linalg.norm(v):.3f}")
    print(f"Horizontal bar descriptor norm: {np.linalg.norm(h):.3f}")
    print(f"Euclidean distance between the two descriptors: {dist:.3f}")
    print(f"Descriptors differ: {dist > 0.5} (vertical edges vs horizontal edges)")
    blank = hog.compute(np.zeros((32, 32)))
    print(f"Blank image descriptor norm: {np.linalg.norm(blank):.3f} (no gradients, all zeros)")


if __name__ == "__main__":
    main()
