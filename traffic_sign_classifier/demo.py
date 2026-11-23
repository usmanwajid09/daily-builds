"""Demo: classify synthetic traffic signs with kNN (k=3).

Generates 60 variants of each of the 6 sign classes for training and
30 fresh variants per class for testing (different seed), trains a
:class:`traffic_sign_classifier.TrafficSignClassifier`, and prints
overall accuracy plus per-class accuracy and a few example predictions.
All images are synthetic, generated in code -- no external files.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python traffic_sign_classifier/demo.py`):
# make the repo root importable so `traffic_sign_classifier` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_sign_classifier.signs import (
    CLASSES,
    TrafficSignClassifier,
    accuracy,
    make_dataset,
)

N_TRAIN_PER_CLASS = 60
N_TEST_PER_CLASS = 30


def main():
    train_images, train_labels = make_dataset(N_TRAIN_PER_CLASS, seed=1)
    test_images, test_labels = make_dataset(N_TEST_PER_CLASS, seed=2)
    clf = TrafficSignClassifier(k=3)
    clf.fit(train_images, train_labels)

    overall = accuracy(test_images, test_labels, clf)
    print(
        f"train: {len(train_images)} images "
        f"({N_TRAIN_PER_CLASS} per class), "
        f"test: {len(test_images)} images "
        f"({N_TEST_PER_CLASS} per class)"
    )
    print(
        f"overall accuracy: {overall:.3f} "
        f"({int(overall * len(test_images))}/{len(test_images)} correct)"
    )
    for sign_type in CLASSES:
        subset = [
            (img, lab)
            for img, lab in zip(test_images, test_labels)
            if lab == sign_type
        ]
        imgs = [img for img, _ in subset]
        labs = [lab for _, lab in subset]
        acc = accuracy(imgs, labs, clf)
        print(f"  {sign_type:12s}: accuracy {acc:.2f}")

    print("example predictions:")
    rng = np.random.default_rng(99)
    for sign_type in ("stop", "warning", "go"):
        from traffic_sign_classifier.signs import make_sign

        img = make_sign(sign_type, rng=rng)
        print(f"  true={sign_type:12s} predicted={clf.predict_one(img)}")
    return overall


if __name__ == "__main__":
    main()
