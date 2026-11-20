"""Demo: classify synthetic hand gestures with Hu moments + kNN.

Generates 30 training and 15 test gestures per class (circle, square,
triangle, star) with random scale, position and rotation, extracts
three invariant features (log-Hu h1, radial spread, solidity), fits a
kNN classifier (k=3), and prints overall and per-class test accuracy
plus a vote-share example.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gesture_classifier.model import (
    GESTURES,
    GestureClassifier,
    accuracy,
    extract_features,
    make_gesture,
)

N_TRAIN = 30
N_TEST = 15
SEED = 11


def _build_split(rng, n_per_class):
    images, labels = [], []
    for name in GESTURES:
        for _ in range(n_per_class):
            images.append(make_gesture(name, rng=rng))
            labels.append(name)
    return images, np.array(labels)


def main():
    rng = np.random.default_rng(SEED)
    train_images, train_labels = _build_split(rng, N_TRAIN)
    test_images, test_labels = _build_split(rng, N_TEST)

    clf = GestureClassifier(k=3).fit(
        extract_features(train_images), train_labels
    )
    overall = accuracy(clf, test_images, test_labels)
    print(f"train: {len(train_images)} gestures, test: {len(test_images)} "
          f"gestures ({N_TRAIN}/{N_TEST} per class, k=3)")
    print(f"overall test accuracy: {overall:.3f} "
          f"({int(overall * len(test_images))}/{len(test_images)} correct)")

    preds = clf.predict(extract_features(test_images))
    for name in GESTURES:
        mask = test_labels == name
        acc = float(np.mean(preds[mask] == test_labels[mask]))
        print(f"  {name:9s}: accuracy {acc:.2f}")
    sample = test_images[0]
    proba = clf.predict_proba(extract_features([sample]))[0]
    print(f"example: true={test_labels[0]}, predicted={clf.classify(sample)}, "
          f"vote shares={dict(zip(clf.classes_, np.round(proba, 2)))}")
    return overall


if __name__ == "__main__":
    main()
