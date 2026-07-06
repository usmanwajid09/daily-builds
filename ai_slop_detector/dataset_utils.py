"""Shared dataset loading/splitting helpers used by train.py, evaluate.py,
and demo.py.

Pulled out of train.py in milestone 2 so evaluate.py doesn't have to
duplicate (and risk drifting from) the same loading/label-validation/split
logic -- a real, if small, refactor rather than copy-paste.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from ai_slop_detector.features import extract_features

DATA_PATH = Path(__file__).parent / "data" / "samples.csv"
HARD_EXAMPLES_PATH = Path(__file__).parent / "data" / "hard_examples.csv"

LABEL_TO_INT = {"human": 0, "ai": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def load_dataset(path: Path = DATA_PATH) -> list[tuple[str, str]]:
    """Loads a two-column (text,label) CSV. Raises on missing/unknown labels
    rather than silently dropping bad rows, since a silent drop would make
    "how many samples did we actually train on" a mystery.
    """
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["text"], row["label"]) for row in reader]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    bad_labels = {label for _, label in rows if label not in LABEL_TO_INT}
    if bad_labels:
        raise ValueError(f"Unknown label(s) in dataset: {bad_labels}")
    return rows


def stratified_split(
    rows: list[tuple[str, str]], test_size: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Splits `rows` into (train, test), preserving each label's proportion.

    A plain random split on a small, evenly-labeled dataset can still land
    on a skewed test set by chance; stratifying avoids that and gives a
    more stable read on precision/recall per class.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)

    train, test = [], []
    for label, group in by_label.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_k_folds(
    rows: list[tuple[str, str]], k: int, seed: int
) -> list[tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    """Returns a list of k (train, test) splits, each fold's test set
    stratified by label and the k test sets partitioning the full dataset
    (every row appears in exactly one fold's test set).

    Used by evaluate.py to get a mean/std accuracy across folds instead of
    trusting a single train/test split, which is exactly the kind of thing
    milestone 1's self-review flagged as missing (REVIEW.md: "No
    cross-validation").
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    rng = random.Random(seed)
    by_label: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)

    # Assign each label's shuffled rows round-robin into k buckets so every
    # bucket gets a near-equal share of each class.
    buckets: list[list[tuple[str, str]]] = [[] for _ in range(k)]
    for label, group in by_label.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        for i, row in enumerate(shuffled):
            buckets[i % k].append(row)

    folds = []
    for i in range(k):
        test_fold = buckets[i]
        train_fold = [row for j, bucket in enumerate(buckets) if j != i for row in bucket]
        folds.append((train_fold, test_fold))
    return folds


def rows_to_xy(rows: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([extract_features(text).to_array() for text, _ in rows])
    y = np.array([LABEL_TO_INT[label] for _, label in rows])
    return X, y
