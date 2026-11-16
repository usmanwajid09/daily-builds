"""Demo: classify synthetic textures with LBP features + kNN.

Generates 30 training and 15 test images per class (stripes,
checkerboard, noise, dots), extracts 256-bin LBP histograms, fits a
kNN classifier (k=3), and prints overall and per-class test accuracy.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python texture_classifier/demo.py`):
# make the repo root importable so `texture_classifier` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from texture_classifier.textures import (
    CLASSES,
    TextureKNN,
    accuracy,
    extract_features,
    make_texture,
)

N_TRAIN = 30
N_TEST = 15
SEED = 7


def _build_split(rng, n_per_class):
    images, labels = [], []
    for name in CLASSES:
        for _ in range(n_per_class):
            images.append(make_texture(name, rng=rng))
            labels.append(name)
    return images, np.array(labels)


def main():
    rng = np.random.default_rng(SEED)
    train_images, train_labels = _build_split(rng, N_TRAIN)
    test_images, test_labels = _build_split(rng, N_TEST)

    clf = TextureKNN(k=3).fit(
        extract_features(train_images), train_labels
    )
    overall = accuracy(clf, test_images, test_labels)
    print(f"train: {len(train_images)} images, test: {len(test_images)} images "
          f"({N_TRAIN}/{N_TEST} per class, k=3)")
    print(f"overall test accuracy: {overall:.3f} "
          f"({int(overall * len(test_images))}/{len(test_images)} correct)")

    preds = clf.predict(extract_features(test_images))
    for name in CLASSES:
        mask = test_labels == name
        acc = float(np.mean(preds[mask] == test_labels[mask]))
        print(f"  {name:12s}: accuracy {acc:.2f}")
    return overall


if __name__ == "__main__":
    main()
