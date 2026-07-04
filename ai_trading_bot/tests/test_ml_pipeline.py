import numpy as np
import pytest

from ai_trading_bot.data import generate_synthetic_ohlcv
from ai_trading_bot.features import build_dataset
from ai_trading_bot.ml_pipeline import (
    classification_metrics,
    signal_from_predictions,
    split_dataset,
    train_direction_model,
)


def test_split_dataset_is_chronological_not_shuffled():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)
    valid_idx = np.arange(100, 110)  # arbitrary original indices
    X_train, y_train, train_idx, X_test, y_test, test_idx = split_dataset(
        X, y, valid_idx, train_frac=0.7
    )
    assert len(X_train) == 7 and len(X_test) == 3
    # train comes strictly before test in original index space
    assert train_idx[-1] < test_idx[0]
    np.testing.assert_array_equal(train_idx, valid_idx[:7])
    np.testing.assert_array_equal(test_idx, valid_idx[7:])


def test_split_dataset_rejects_bad_train_frac():
    X = np.zeros((10, 2))
    y = np.zeros(10)
    idx = np.arange(10)
    with pytest.raises(ValueError):
        split_dataset(X, y, idx, train_frac=0.0)
    with pytest.raises(ValueError):
        split_dataset(X, y, idx, train_frac=1.0)


def test_split_dataset_rejects_too_few_rows():
    X = np.zeros((1, 2))
    y = np.zeros(1)
    idx = np.arange(1)
    with pytest.raises(ValueError):
        split_dataset(X, y, idx, train_frac=0.7)


def test_train_direction_model_end_to_end_shapes():
    s = generate_synthetic_ohlcv(n_days=400, seed=5)
    result = train_direction_model(s.close, s.volume, train_frac=0.7, n_iters=200)
    total_valid = len(build_dataset(s.close, s.volume)[0])
    assert len(result.train_idx) + len(result.test_idx) == total_valid
    assert len(result.test_pred) == len(result.test_idx) == len(result.y_test)
    assert set(np.unique(result.test_pred)).issubset({0, 1})
    assert np.all((result.test_proba >= 0) & (result.test_proba <= 1))
    # test indices must come strictly after train indices (no shuffling)
    assert result.train_idx[-1] < result.test_idx[0]


def test_classification_metrics_hand_computed():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    # tp=2 (idx0,4), fp=1 (idx3), tn=1 (idx2), fn=1 (idx1)
    m = classification_metrics(y_true, y_pred)
    assert m["confusion_matrix"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_classification_metrics_no_predicted_positives_precision_zero():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == 0.0  # no positive predictions -> defined as 0, not nan
    assert m["recall"] == 0.0


def test_classification_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        classification_metrics(np.array([1, 0]), np.array([1]))


def test_signal_from_predictions_only_sets_test_bars():
    n_bars = 10
    test_idx = np.array([6, 7, 8, 9])
    test_pred = np.array([1, 0, 1, 1])
    signal = signal_from_predictions(n_bars, test_idx, test_pred)
    np.testing.assert_array_equal(signal[:6], 0)
    np.testing.assert_array_equal(signal[6:], [1, 0, 1, 1])


def test_signal_from_predictions_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        signal_from_predictions(10, np.array([1, 2, 3]), np.array([1, 0]))
