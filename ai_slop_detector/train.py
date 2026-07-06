"""CLI: load the labeled dataset, extract stylometric features, train the
from-scratch logistic regression classifier, and report evaluation metrics.

Usage:
    python -m ai_slop_detector.train
    python -m ai_slop_detector.train --test-size 0.3 --epochs 3000

For a more rigorous evaluation (k-fold cross-validation + a held-out
hard/adversarial example set), see `python -m ai_slop_detector.evaluate`.
This script keeps the simple single-split path since it's also what
produces the `trained_model.json` that `demo.py` loads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_slop_detector.classifier import (
    LogisticRegression,
    StandardScaler,
    classification_metrics,
)
from ai_slop_detector.dataset_utils import (
    LABEL_TO_INT,
    load_dataset,
    rows_to_xy,
    stratified_split,
)
from ai_slop_detector.features import FEATURE_NAMES

MODEL_PATH = Path(__file__).parent / "trained_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_dataset()
    train_rows, test_rows = stratified_split(rows, args.test_size, args.seed)
    print(f"Loaded {len(rows)} samples -> {len(train_rows)} train / {len(test_rows)} test")

    X_train, y_train = rows_to_xy(train_rows)
    X_test, y_test = rows_to_xy(test_rows)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(lr=args.lr, epochs=args.epochs, l2=args.l2)
    model.fit(X_train_scaled, y_train)

    train_metrics = classification_metrics(y_train, model.predict(X_train_scaled))
    test_metrics = classification_metrics(y_test, model.predict(X_test_scaled))

    print("\n--- Train metrics ---")
    _print_metrics(train_metrics)
    print("\n--- Test metrics ---")
    _print_metrics(test_metrics)
    print(
        "\nNote: this is a single stratified split on the synthetic "
        "template-based dataset -- see `python -m ai_slop_detector.evaluate` "
        "for k-fold cross-validation and an out-of-distribution hard-example "
        "check, which paint a less flattering (and more honest) picture."
    )

    print("\n--- Feature weights (standardized scale; sign shows ai(+)/human(-) pull) ---")
    for name, weight in sorted(
        zip(FEATURE_NAMES, model.weights), key=lambda t: -abs(t[1])
    ):
        print(f"  {name:<38s} {weight:+.3f}")

    _save_model(model, scaler)
    print(f"\nSaved trained weights + scaler stats to {MODEL_PATH}")


def _print_metrics(metrics: dict) -> None:
    print(
        f"  accuracy={metrics['accuracy']:.3f}  precision={metrics['precision']:.3f}  "
        f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}"
    )
    print(
        f"  confusion matrix: tp={metrics['tp']} tn={metrics['tn']} "
        f"fp={metrics['fp']} fn={metrics['fn']}"
    )


def _save_model(model: LogisticRegression, scaler: StandardScaler) -> None:
    payload = {
        "feature_names": FEATURE_NAMES,
        "weights": model.weights.tolist(),
        "bias": model.bias,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.std_.tolist(),
        "label_to_int": LABEL_TO_INT,
    }
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
