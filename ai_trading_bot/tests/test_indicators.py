import numpy as np
import pytest

from ai_trading_bot.indicators import ema, macd, rsi, sma


def test_sma_basic():
    values = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    out = sma(values, window=3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [2.0, 3.0, 4.0, 5.0])


def test_sma_window_larger_than_series_is_all_nan():
    out = sma(np.array([1.0, 2.0]), window=5)
    assert np.all(np.isnan(out))


def test_sma_rejects_bad_window():
    with pytest.raises(ValueError):
        sma(np.array([1.0, 2.0]), window=0)


def test_ema_matches_hand_computed_values():
    # window=3 -> alpha = 2/4 = 0.5
    values = np.array([1, 2, 3, 4, 5], dtype=float)
    out = ema(values, window=3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    seed = (1 + 2 + 3) / 3  # = 2.0
    e3 = 0.5 * 4 + 0.5 * seed  # = 3.0
    e4 = 0.5 * 5 + 0.5 * e3  # = 4.0
    np.testing.assert_allclose(out[2:], [seed, e3, e4])


def test_ema_converges_toward_constant_series():
    values = np.full(50, 10.0)
    out = ema(values, window=5)
    assert np.isnan(out[:4]).all()
    np.testing.assert_allclose(out[4:], 10.0)


def test_rsi_all_gains_is_100():
    values = np.arange(1, 20, dtype=float)  # strictly increasing
    out = rsi(values, window=14)
    valid = out[~np.isnan(out)]
    assert np.allclose(valid, 100.0)


def test_rsi_all_losses_is_0():
    values = np.arange(20, 1, -1, dtype=float)  # strictly decreasing
    out = rsi(values, window=14)
    valid = out[~np.isnan(out)]
    assert np.allclose(valid, 0.0)


def test_rsi_bounded_0_100_on_noisy_data():
    rng = np.random.default_rng(0)
    values = 100 + np.cumsum(rng.normal(0, 1, 200))
    out = rsi(values, window=14)
    valid = out[~np.isnan(out)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


def test_rsi_warmup_length():
    values = np.arange(1, 30, dtype=float)
    out = rsi(values, window=14)
    assert np.isnan(out[:14]).all()
    assert not np.isnan(out[14])


def test_macd_shapes_and_warmup():
    rng = np.random.default_rng(1)
    values = 100 + np.cumsum(rng.normal(0, 1, 100))
    macd_line, signal_line, hist = macd(values, fast=12, slow=26, signal=9)
    assert len(macd_line) == len(signal_line) == len(hist) == 100
    # macd_line warms up at index slow-1 = 25
    assert np.isnan(macd_line[:25]).all()
    assert not np.isnan(macd_line[25])
    # signal line warms up `signal` points after that
    assert np.isnan(signal_line[: 25 + 9 - 1]).all()
    assert not np.isnan(signal_line[25 + 9 - 1])
    np.testing.assert_allclose(hist[~np.isnan(hist)], (macd_line - signal_line)[~np.isnan(hist)])


def test_macd_rejects_fast_ge_slow():
    with pytest.raises(ValueError):
        macd(np.arange(50, dtype=float), fast=26, slow=12)


def test_macd_zero_on_flat_series():
    values = np.full(60, 50.0)
    macd_line, signal_line, hist = macd(values, fast=12, slow=26, signal=9)
    valid = ~np.isnan(macd_line)
    np.testing.assert_allclose(macd_line[valid], 0.0, atol=1e-9)
