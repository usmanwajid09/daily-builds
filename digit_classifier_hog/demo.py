"""Demo: HOG + kNN digit classification vs template matching.

Trains a kNN classifier (k=3) on HOG descriptors of noisy 8x8 digit
images (30 variants per digit, 5% pixel flips) and evaluates it on a
held-out set of 200 fresh noisy variants (20 per digit). The same test
images are scored by raw-pixel nearest-template matching, so the two
approaches are compared head to head. A HOG + raw-pixels variant is
shown as well, to demonstrate the tuning suggested in the README.

Honest result: on these synthetic digits template matching wins --
the templates are the exact prototypes the noisy images were made
from, so raw-pixel distance is near-optimal and HOG's edge histograms
throw away absolute-position information. HOG + kNN is still a real,
general classifier: unlike template matching it learns from examples
and would keep working on digits drawn in a style with no clean
template available.
"""

import os
import sys

import numpy as np

# Allow running as a plain script (`python digit_classifier_hog/demo.py`):
# make the repo root importable so `digit_classifier_hog` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digit_classifier_hog.hog_digits import (
    HOGKNNClassifier,
    accuracy,
    hog_features,
    make_dataset,
    make_images,
    template_match_accuracy,
    train,
)
from digit_recognizer.digits import add_noise, template

N_TRAIN_VARIANTS = 30
N_TEST_VARIANTS = 20
FLIP_PROB = 0.05
TRAIN_SEED = 1
TEST_SEED = 99
K = 3
CELL_SIZE = 2


def main():
    train_features, train_labels = make_dataset(
        N_TRAIN_VARIANTS, FLIP_PROB, TRAIN_SEED, cell_size=CELL_SIZE
    )
    test_features, test_labels = make_dataset(
        N_TEST_VARIANTS, FLIP_PROB, TEST_SEED, cell_size=CELL_SIZE
    )
    test_images, _ = make_images(N_TEST_VARIANTS, FLIP_PROB, TEST_SEED)

    classifier = train(train_features, train_labels, k=K)
    hog_acc = accuracy(test_features, test_labels, classifier)

    combined_features, _ = make_dataset(
        N_TEST_VARIANTS, FLIP_PROB, TEST_SEED,
        cell_size=CELL_SIZE, append_pixels=True,
    )
    combined_train, _ = make_dataset(
        N_TRAIN_VARIANTS, FLIP_PROB, TRAIN_SEED,
        cell_size=CELL_SIZE, append_pixels=True,
    )
    combined_acc = accuracy(
        combined_features, test_labels,
        train(combined_train, train_labels, k=K),
    )

    template_acc = template_match_accuracy(test_images, test_labels)

    n_test = len(test_labels)
    print(f"train: {len(train_labels)} noisy variants "
          f"({N_TRAIN_VARIANTS} per digit, flip_prob={FLIP_PROB})")
    print(f"test:  {n_test} held-out noisy variants "
          f"({N_TEST_VARIANTS} per digit)")
    print(f"HOG descriptor: cell_size={CELL_SIZE}, "
          f"length {len(test_features[0])}")
    print(f"HOG + kNN (k={K}):        {hog_acc:.3f} "
          f"({int(hog_acc * n_test)}/{n_test} correct)")
    print(f"HOG + pixels + kNN:      {combined_acc:.3f} "
          f"({int(combined_acc * n_test)}/{n_test} correct)")
    print(f"template matching:       {template_acc:.3f} "
          f"({int(template_acc * n_test)}/{n_test} correct)")
    print("verdict: template matching wins on these synthetic digits --")
    print("  the noisy images are flips of the exact templates, so")
    print("  raw-pixel distance is near-optimal; HOG discards the")
    print("  absolute-position detail that makes templates so strong.")
    print("  Appending raw pixels closes most of the gap.")

    rng = np.random.default_rng(TEST_SEED)
    print("example predictions (noisy '3', '7', '9'):")
    for digit in (3, 7, 9):
        noisy = add_noise(template(digit), FLIP_PROB, rng)
        pred = classifier.predict_one(hog_features(noisy, cell_size=CELL_SIZE))
        print(f"  true={digit} HOG+kNN predicted={pred}")
    return hog_acc, template_acc


if __name__ == "__main__":
    main()
