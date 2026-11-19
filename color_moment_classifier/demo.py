"""Demo: train a colour-moment + kNN classifier on synthetic colour scenes.

Builds 15 synthetic images per class (sunset / forest / ocean) for
training and 9 fresh ones per class for testing, all with fixed seeds,
then prints per-class and overall test accuracy and the 9-dim feature
vector of one sunset image.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from color_moment_classifier.moments import (
    ColorMomentClassifier,
    extract_features,
    generate_scene,
)

import numpy as np

KINDS = ["sunset", "forest", "ocean"]


def make_dataset(n_train=15, n_test=9, size=32):
    """Build a seeded synthetic train/test split of colour scenes.

    Returns (train_images, train_labels, test_images, test_labels).
    Training seeds are 0..n_train-1 per kind; test seeds are
    100..100+n_test-1 per kind, never seen in training.
    """
    train_images, train_labels = [], []
    for kind in KINDS:
        for seed in range(n_train):
            train_images.append(generate_scene(kind, size=size, seed=seed))
            train_labels.append(kind)
    test_images, test_labels = [], []
    for kind in KINDS:
        for seed in range(100, 100 + n_test):
            test_images.append(generate_scene(kind, size=size, seed=seed))
            test_labels.append(kind)
    return train_images, train_labels, test_images, test_labels


def make_features(images):
    """Return the (n, 9) colour-moment feature matrix for a list of images."""
    return np.array([extract_features(img) for img in images])


def main():
    train_images, train_labels, test_images, test_labels = make_dataset()
    X_train = make_features(train_images)
    X_test = make_features(test_images)
    print(f"Training set: {len(train_labels)} images "
          f"({X_train.shape[1]}-dim colour-moment features)")
    print(f"Test set: {len(test_labels)} images")

    clf = ColorMomentClassifier(k=3).fit(X_train, np.array(train_labels))
    print(f"Classifier: k={clf.k}, classes={list(clf.classes_)}")

    preds = clf.predict(X_test)
    y_test = np.array(test_labels)
    acc = clf.score(X_test, y_test)
    print(f"Overall test accuracy: {acc:.2%} "
          f"({int((preds == y_test).sum())}/{len(y_test)} correct)")

    # Per-class accuracy.
    for kind in KINDS:
        mask = y_test == kind
        print(f"  {kind:6s}: {np.mean(preds[mask] == kind):.2%} "
              f"({int((preds[mask] == kind).sum())}/{int(mask.sum())} correct)")

    # Example predictions: one test image per kind.
    print("Example predictions:")
    for i in [0, 9, 18]:
        print(f"  true={test_labels[i]:6s} predicted={preds[i]:6s}")

    # The 9-dim feature vector of one sunset image, rounded to 3 decimals.
    sunset = generate_scene("sunset", size=32, seed=0)
    feats = extract_features(sunset)
    print("Sunset feature vector (mean, std, skew per R/G/B channel):")
    print(f"  {np.round(feats, 3).tolist()}")


if __name__ == "__main__":
    main()
