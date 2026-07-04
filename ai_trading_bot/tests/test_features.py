import numpy as np
import pytest

from ai_trading_bot.data import generate_synthetic_ohlcv
from ai_trading_bot.features import FEATURE_NAMES, build_dataset, build_features, build_labels


def test_build_features_shape():
    s = generate_synthetic_ohlcv(n_days=200, seed=1)
    feats = build_features(s.close, s.volume)
    assert feats.shape == (200, len(FEATURE_NAMES))


def test_build_features_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_features(np.arange(10, dtype=float), np.arange(5, dtype=float))


def test_build_features_warmup_rows_are_nan():
    s = generate_synthetic_ohlcv(n_days=100, seed=2)
    feats = build_features(s.close, s.volume)
    # sma20_ratio (col 2) needs 20 bars of warmup -> row 0 must be NaN
    assert np.isnan(feats[0, 2])
    # by day 60, everything should have warmed up
    assert not np.any(np.isnan(feats[60]))


def test_build_labels_matches_hand_computation():
    close = np.array([100.0, 105.0, 100.0, 100.0])
    labels = build_labels(close)
    # day0->1 up (1), day1->2 down (0), day2->3 flat (0, not strictly greater)
    np.testing.assert_array_equal(labels[:3], [1.0, 0.0, 0.0])
    assert np.isnan(labels[-1])  # no next day for the last bar


def test_build_dataset_drops_nan_rows_and_tracks_original_indices():
    s = generate_synthetic_ohlcv(n_days=100, seed=3)
    X, y, valid_idx = build_dataset(s.close, s.volume)
    assert len(X) == len(y) == len(valid_idx)
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))
    assert set(np.unique(y)).issubset({0.0, 1.0})
    # valid_idx should be strictly increasing and within bounds
    assert np.all(np.diff(valid_idx) > 0)
    assert valid_idx[-1] < len(s.close) - 1  # last bar (no label) excluded
    assert valid_idx[0] >= 20  # sma20 warmup excludes early bars


def test_build_dataset_rejects_too_short_series():
    with pytest.raises(ValueError):
        build_features(np.array([1.0]), np.array([1.0]))
