"""Rigorous evaluation: stratified k-fold cross-validation on the main
synthetic dataset, plus an out-of-distribution check against a small,
hand-written "hard examples" set.

This is the direct follow-up to milestone 1's self-review finding
(REVIEW.md): a single train/test split on samples.csv reports 100%/100%
because the two template families are trivially separable, which is a
statement about the dataset, not the technique. This script reports:

  1. Mean +/- stdev accuracy/F1 across k stratified folds of samples.csv
     (still the same synthetic dataset, so still optimistic, but at least
     not a single lucky/unlucky split).
  2. Accuracy on data/hard_examples.csv -- 20 hand-written sentences
     deliberately built to violate the main dataset's assumptions (formal
     human writing, buzzwordy human writing, casual/contraction-heavy
     "AI" writing) -- trained on all of samples.csv, evaluated only on
     the untouched hard set.

Usage:
    python -m ai_slop_detector.evaluate
    python -m ai_slop_detector.evaluate --folds 10
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from ai_slop_detector.classifier import (
    LogisticRegression,
    StandardScaler,
    classification_report,
    format_classification_report,
)
from ai_slop_detector.dataset_utils import (
    HARD_EXAMPLES_PATH,
    LABEL_TO_INT,
    load_dataset,
    rows_to_xy,
    stratified_k_folds,
)

LABEL_NAMES = {0: "human", 1: "ai"}


def load_hard_examples(path=HARD_EXAMPLES_PATH) -> list[tuple[str, str, str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["text"], row["label"], row.get("note", "")) for row in reader]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    bad_labels = {label for _, label, _ in rows if label not in LABEL_TO_INT}
    if bad_labels:
        raise ValueError(f"Unknown label(s) in hard examples: {bad_labels}")
    return rows


def run_cross_validation(
    rows: list[tuple[str, str]], k: int, epochs: int, lr: float, l2: float, seed: int
) -> list[dict]:
    folds = stratified_k_folds(rows, k=k, seed=seed)
    fold_reports = []
    for i, (train_rows, test_rows) in enumerate(folds, start=1):
        X_train, y_train = rows_to_xy(train_rows)
        X_test, y_test = rows_to_xy(test_rows)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(lr=lr, epochs=epochs, l2=l2)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        report = classification_report(y_test, preds)
        fold_reports.append(report)
        print(f"\n--- Fold {i}/{k} (n_test={len(test_rows)}) ---")
        print(format_classification_report(report, LABEL_NAMES))
    return fold_reports


def summarize_folds(fold_reports: list[dict]) -> None:
    accuracies = [r["accuracy"]["value"] for r in fold_reports]
    macro_f1s = [r["macro_avg"]["f1"] for r in fold_reports]
    print(f"\n=== Cross-validation summary over {len(fold_reports)} folds ===")
    print(f"  accuracy: mean={np.mean(accuracies):.3f}  stdev={np.std(accuracies):.3f}")
    print(f"  macro F1: mean={np.mean(macro_f1s):.3f}  stdev={np.std(macro_f1s):.3f}")


def run_hard_example_check(
    train_rows: list[tuple[str, str]], epochs: int, lr: float, l2: float
) -> None:
    hard_rows = load_hard_examples()
    X_train, y_train = rows_to_xy(train_rows)
    X_hard, y_hard = rows_to_xy([(text, label) for text, label, _ in hard_rows])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_hard_scaled = scaler.transform(X_hard)

    model = LogisticRegression(lr=lr, epochs=epochs, l2=l2)
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_hard_scaled)
    preds = (probs >= 0.5).astype(int)

    print(f"\n=== Hard/out-of-distribution example check (n={len(hard_rows)}) ===")
    report = classification_report(y_hard, preds)
    print(format_classification_report(report, LABEL_NAMES))

    print("\nPer-example predictions (* = wrong):")
    for (text, label, note), prob, pred in zip(hard_rows, probs, preds):
        pred_label = LABEL_NAMES[pred]
        marker = "*" if pred_label != label else " "
        short_text = text if len(text) <= 60 else text[:57] + "..."
        print(f"  {marker} true={label:<6} pred={pred_label:<6} p_ai={prob:.3f}  {short_text}")
        if marker == "*":
            print(f"      why it's hard: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_dataset()
    print(f"Loaded {len(rows)} samples from samples.csv for {args.folds}-fold cross-validation")
    fold_reports = run_cross_validation(rows, args.folds, args.epochs, args.lr, args.l2, args.seed)
    summarize_folds(fold_reports)

    run_hard_example_check(rows, args.epochs, args.lr, args.l2)

    print(
        "\nRead together: cross-validation on samples.csv measures whether the "
        "model reliably separates the two *template families*; the hard-example "
        "check measures whether that generalizes past them. Expect the first "
        "number to look great and the second to look much rougher -- that gap "
        "is the honest headline result of this project, not a bug to hide."
    )


if __name__ == "__main__":
    main()
