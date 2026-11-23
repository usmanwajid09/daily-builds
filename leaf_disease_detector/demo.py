"""Demo: classify synthetic healthy / spotted / yellowing leaves.

Generates 30 leaves per class for training and 15 per class for testing,
trains the kNN classifier, and prints overall accuracy, per-class
accuracy, and a few example predictions.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python leaf_disease_detector/demo.py`):
# make the repo root importable so `leaf_disease_detector` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leaf_disease_detector.leaves import (
    CLASSES,
    LeafDiseaseClassifier,
    accuracy,
    make_dataset,
    make_leaf,
)

N_TRAIN = 30
N_TEST = 15
TRAIN_SEED = 7
TEST_SEED = 99


def main():
    train_images, train_labels = make_dataset(N_TRAIN, seed=TRAIN_SEED)
    test_images, test_labels = make_dataset(N_TEST, seed=TEST_SEED)

    clf = LeafDiseaseClassifier(k=3)
    clf.fit(train_images, train_labels)

    overall = accuracy(test_images, test_labels, clf)
    print(f"train: {len(train_images)} leaves "
          f"({N_TRAIN} per class), test: {len(test_images)} leaves "
          f"({N_TEST} per class), k=3")
    print(f"overall accuracy: {overall:.3f} "
          f"({int(overall * len(test_images))}/{len(test_images)} correct)")

    for label in CLASSES:
        subset = [img for img, lab in zip(test_images, test_labels)
                  if lab == label]
        acc = accuracy(subset, [label] * len(subset), clf)
        print(f"  {label}: accuracy {acc:.2f}")

    print("example predictions:")
    rng = np.random.default_rng(123)
    for label in CLASSES:
        leaf = make_leaf(label, rng=rng)
        print(f"  true={label} predicted={clf.predict(leaf)}")
    return overall


if __name__ == "__main__":
    main()
