import numpy as np
import pytest

from ai_slop_detector.dataset_utils import (
    load_dataset,
    rows_to_xy,
    stratified_k_folds,
    stratified_split,
)


def _toy_rows(n_human=10, n_ai=10):
    rows = [(f"human text {i}", "human") for i in range(n_human)]
    rows += [(f"ai text {i}", "ai") for i in range(n_ai)]
    return rows


def test_stratified_split_preserves_label_proportions():
    rows = _toy_rows(20, 20)
    train, test = stratified_split(rows, test_size=0.25, seed=1)
    train_ai = sum(1 for _, label in train if label == "ai")
    test_ai = sum(1 for _, label in test if label == "ai")
    assert test_ai == 5  # 25% of 20
    assert train_ai == 15
    assert len(train) + len(test) == len(rows)


def test_stratified_split_no_overlap():
    rows = _toy_rows(15, 15)
    train, test = stratified_split(rows, test_size=0.3, seed=7)
    assert set(train).isdisjoint(set(test))


def test_stratified_k_folds_partitions_full_dataset():
    rows = _toy_rows(20, 20)
    folds = stratified_k_folds(rows, k=5, seed=3)
    assert len(folds) == 5

    all_test_rows = []
    for train_fold, test_fold in folds:
        assert set(train_fold).isdisjoint(set(test_fold))
        all_test_rows.extend(test_fold)

    # Every row appears in exactly one fold's test set.
    assert sorted(all_test_rows) == sorted(rows)


def test_stratified_k_folds_each_fold_has_both_labels():
    rows = _toy_rows(20, 20)
    folds = stratified_k_folds(rows, k=5, seed=3)
    for _, test_fold in folds:
        labels = {label for _, label in test_fold}
        assert labels == {"human", "ai"}


def test_stratified_k_folds_rejects_k_less_than_2():
    with pytest.raises(ValueError):
        stratified_k_folds(_toy_rows(), k=1, seed=0)


def test_rows_to_xy_shapes():
    rows = [("I don't think so, honestly.", "human"), ("Furthermore, it is important to note this.", "ai")]
    X, y = rows_to_xy(rows)
    assert X.shape[0] == 2
    assert list(y) == [0, 1]


def test_load_dataset_rejects_unknown_labels(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("text,label\nhello world,robot\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(bad_csv)


def test_load_dataset_rejects_empty_file(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("text,label\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(empty_csv)
