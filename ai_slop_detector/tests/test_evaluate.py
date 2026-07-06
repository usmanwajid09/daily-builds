import csv

import numpy as np
import pytest

from ai_slop_detector.classifier import classification_report, format_classification_report
from ai_slop_detector.dataset_utils import HARD_EXAMPLES_PATH
from ai_slop_detector.evaluate import load_hard_examples


def test_hard_examples_csv_exists_and_is_well_formed():
    assert HARD_EXAMPLES_PATH.exists(), "hard_examples.csv must be committed"
    rows = load_hard_examples()
    assert len(rows) > 0
    for text, label, note in rows:
        assert label in {"human", "ai"}
        assert text.strip() != ""
        assert note.strip() != "", "every hard example should document why it's hard"


def test_hard_examples_has_both_labels_represented():
    rows = load_hard_examples()
    labels = {label for _, label, _ in rows}
    assert labels == {"human", "ai"}


def test_hard_examples_no_duplicate_texts():
    rows = load_hard_examples()
    texts = [text for text, _, _ in rows]
    assert len(texts) == len(set(texts))


def test_classification_report_matches_sklearn_style_values():
    # Hand-computed expected values for a small, known confusion matrix.
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1])
    report = classification_report(y_true, y_pred)

    # class 0 ("human"): tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3
    assert report["0"]["precision"] == pytest.approx(2 / 3)
    assert report["0"]["recall"] == pytest.approx(2 / 3)
    assert report["0"]["support"] == 3

    # class 1 ("ai"): tp=3, fp=1, fn=1 -> precision=3/4, recall=3/4
    assert report["1"]["precision"] == pytest.approx(3 / 4)
    assert report["1"]["recall"] == pytest.approx(3 / 4)
    assert report["1"]["support"] == 4

    assert report["accuracy"]["value"] == pytest.approx(5 / 7)


def test_classification_report_handles_a_class_with_zero_support():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 0, 1])
    report = classification_report(y_true, y_pred)
    assert report["0"]["support"] == 0
    assert report["0"]["precision"] == 0.0  # tp+fp == 0 -> defined as 0, not NaN


def test_format_classification_report_is_readable_text():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    report = classification_report(y_true, y_pred)
    text = format_classification_report(report, {0: "human", 1: "ai"})
    assert "human" in text
    assert "ai" in text
    assert "accuracy" in text


def test_cli_rejects_folds_less_than_2_with_clean_error(tmp_path):
    # Regression test: previously `--folds 1` bubbled a multi-frame
    # ValueError traceback from dataset_utils.stratified_k_folds instead of
    # a clean CLI error. Should now exit non-zero via argparse's error()
    # with a message naming the actual flag, not a traceback.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "ai_slop_detector.evaluate", "--folds", "1"],
        capture_output=True,
        text=True,
        cwd=str(HARD_EXAMPLES_PATH.parent.parent.parent),
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "--folds must be >= 2" in result.stderr
