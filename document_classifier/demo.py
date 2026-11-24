"""Demo: train a nearest-centroid document classifier on synthetic docs
and report per-class + overall accuracy on held-out seeds.

Run from the repo root:
    python document_classifier/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_classifier import (  # noqa: E402
    CLASSES,
    accuracy,
    confusion_matrix,
    make_document,
    train_classifier,
)

N_TRAIN = 20
N_TEST = 20


def main() -> None:
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    for cls in CLASSES:
        for seed in range(N_TRAIN):
            train_images.append(make_document(cls, seed))
            train_labels.append(cls)
        for seed in range(1000, 1000 + N_TEST):
            test_images.append(make_document(cls, seed))
            test_labels.append(cls)

    clf = train_classifier(train_images, train_labels)

    print(f"Document classifier — nearest centroid over layout features")
    print(f"Train: {N_TRAIN} docs/class  |  Test: {N_TEST} held-out docs/class")
    print(f"Overall accuracy: {accuracy(clf, test_images, test_labels):.1%}")
    print()
    print("Per-class accuracy:")
    for cls in CLASSES:
        idx = [i for i, lab in enumerate(test_labels) if lab == cls]
        acc = accuracy(clf, [test_images[i] for i in idx],
                       [test_labels[i] for i in idx])
        print(f"  {cls:<8} {acc:.1%}  ({int(acc * len(idx))}/{len(idx)})")
    print()
    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(clf, test_images, test_labels)
    header = "true \\ pred " + "".join(f"{c:>9}" for c in CLASSES)
    print(header)
    for true in CLASSES:
        row = "".join(f"{cm[true][pred]:>9}" for pred in CLASSES)
        print(f"{true:<11}{row}")
    print()
    sample = make_document("invoice", seed=4242)
    print("Example: predict_proba(invoice, seed=4242) ->")
    for cls, prob in clf.predict_proba(sample).items():
        print(f"  {cls:<8} {prob:.3f}")


if __name__ == "__main__":
    main()
