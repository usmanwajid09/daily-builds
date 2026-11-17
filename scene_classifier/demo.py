"""Demo: train a colour-histogram + kNN scene classifier on synthetic scenes.

Builds 15 synthetic images per class (beach / forest / city) for training
and 5 fresh ones per class for testing, all with a fixed seed, then prints
the test accuracy and a few example predictions with vote fractions.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_classifier.scene_classifier import (
    SceneClassifier,
    extract_features,
    generate_scene,
)

import numpy as np


def main():
    rng_seed = 42
    kinds = ["beach", "forest", "city"]

    # Training set: 15 scenes per class, seeds 0-14.
    train_images, train_labels = [], []
    for kind in kinds:
        for seed in range(15):
            train_images.append(generate_scene(kind, seed=seed))
            train_labels.append(kind)
    X_train = np.array([extract_features(img) for img in train_images])
    print(f"Training set: {len(train_labels)} images "
          f"({X_train.shape[1]}-dim histogram features)")

    clf = SceneClassifier(k=3).fit(X_train, np.array(train_labels))
    print(f"Classifier: k={clf.k}, classes={list(clf.classes_)}")

    # Test set: 5 fresh scenes per class, seeds 100-104 (never seen in training).
    test_images, test_labels = [], []
    for kind in kinds:
        for seed in range(100, 105):
            test_images.append(generate_scene(kind, seed=seed))
            test_labels.append(kind)
    X_test = np.array([extract_features(img) for img in test_images])
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    acc = clf.score(X_test, np.array(test_labels))
    print(f"Test accuracy: {acc:.2%} "
          f"({int((preds == np.array(test_labels)).sum())}/{len(test_labels)} correct)")

    # Per-class accuracy.
    for kind in kinds:
        mask = np.array(test_labels) == kind
        print(f"  {kind:6s}: {np.mean(preds[mask] == kind):.2%} "
              f"({int((preds[mask] == kind).sum())}/5 correct)")

    # Example predictions with vote fractions.
    print("Example predictions:")
    for i in [0, 5, 10]:  # one beach, one forest, one city test image
        votes = {c: round(float(p), 2) for c, p in zip(clf.classes_, proba[i])}
        print(f"  true={test_labels[i]:6s} predicted={preds[i]:6s} votes={votes}")

    # A hand-built image: mostly blue -> should look like a beach/sky scene.
    blue = np.zeros((32, 32, 3))
    blue[:, :] = [0.4, 0.7, 0.95]
    print(f"All-blue image -> {clf.predict(extract_features(blue)[None, :])[0]}")


if __name__ == "__main__":
    main()
