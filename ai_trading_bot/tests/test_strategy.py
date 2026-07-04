import numpy as np
import pytest

from ai_trading_bot.strategy import (
    combined_trend_and_momentum_signal,
    macd_signal,
    rsi_mean_reversion_signal,
    sma_crossover_signal,
)


def test_sma_crossover_signal_is_binary_and_right_length():
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 1, 120))
    sig = sma_crossover_signal(close, fast=5, slow=20)
    assert len(sig) == len(close)
    assert set(np.unique(sig)).issubset({0, 1})
    assert np.all(sig[:19] == 0)  # slow SMA not warmed up yet


def test_sma_crossover_rejects_fast_ge_slow():
    with pytest.raises(ValueError):
        sma_crossover_signal(np.arange(100, dtype=float), fast=20, slow=5)


def test_sma_crossover_uptrend_goes_long():
    close = np.linspace(100, 200, 100)  # strong steady uptrend
    sig = sma_crossover_signal(close, fast=5, slow=20)
    assert sig[-1] == 1


def test_sma_crossover_downtrend_stays_flat():
    close = np.linspace(200, 100, 100)  # strong steady downtrend
    sig = sma_crossover_signal(close, fast=5, slow=20)
    assert sig[-1] == 0


def test_macd_signal_is_binary():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, 120))
    sig = macd_signal(close)
    assert set(np.unique(sig)).issubset({0, 1})
    assert len(sig) == len(close)


def test_rsi_mean_reversion_enters_and_exits():
    # Craft a series that dips (oversold) then rallies hard (overbought).
    dip = np.linspace(100, 60, 30)
    rally = np.linspace(60, 160, 30)
    close = np.concatenate([dip, rally])
    sig = rsi_mean_reversion_signal(close, window=14, oversold=30, overbought=70)
    assert set(np.unique(sig)).issubset({0, 1})
    # should have gone long at some point during/after the dip
    assert sig.sum() > 0


def test_rsi_mean_reversion_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        rsi_mean_reversion_signal(np.arange(50, dtype=float), oversold=80, overbought=20)


def test_combined_signal_is_subset_of_either_alone():
    rng = np.random.default_rng(4)
    close = 100 + np.cumsum(rng.normal(0, 1, 150))
    trend = sma_crossover_signal(close)
    momentum = macd_signal(close)
    combined = combined_trend_and_momentum_signal(close)
    # combined can only be long where BOTH trend and momentum are long
    assert np.all(combined <= trend)
    assert np.all(combined <= momentum)
