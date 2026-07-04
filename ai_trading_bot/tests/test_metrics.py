import numpy as np
import pytest

from ai_trading_bot.metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    summarize,
    total_return,
    win_rate,
)


def test_total_return_basic():
    equity = np.array([1000.0, 1100.0, 1210.0])
    assert total_return(equity) == pytest.approx(0.21)


def test_total_return_rejects_bad_input():
    with pytest.raises(ValueError):
        total_return(np.array([]))
    with pytest.raises(ValueError):
        total_return(np.array([0.0, 100.0]))


def test_cagr_matches_hand_computation():
    # Exactly double over 1 trading year (252 bars) -> CAGR should be ~100%.
    equity = np.empty(253)
    equity[0] = 1000.0
    equity[1:] = np.linspace(1000.0, 2000.0, 252)
    # simpler: geometric doubling
    equity = 1000.0 * 2.0 ** (np.arange(253) / 252)
    result = cagr(equity, periods_per_year=252)
    assert result == pytest.approx(1.0, rel=1e-6)


def test_cagr_rejects_bad_input():
    with pytest.raises(ValueError):
        cagr(np.array([1000.0]))
    with pytest.raises(ValueError):
        cagr(np.array([-1.0, 2.0]))


def test_sharpe_ratio_zero_variance_returns_zero():
    returns = np.zeros(50)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_steady_positive_returns():
    rng = np.random.default_rng(5)
    returns = 0.001 + rng.normal(0, 0.0001, 300)  # small positive drift, low noise
    s = sharpe_ratio(returns)
    assert s > 0


def test_sharpe_ratio_rejects_too_short_series():
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([0.01]))


def test_max_drawdown_basic():
    equity = np.array([100.0, 120.0, 90.0, 95.0, 130.0])
    # peak 120 -> trough 90 => -25%
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_monotonic_up_is_zero():
    equity = np.array([100.0, 110.0, 120.0, 130.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_win_rate_excludes_zero_return_bars():
    returns = np.array([0.01, -0.02, 0.0, 0.03, 0.0, -0.01])
    # nonzero: [0.01, -0.02, 0.03, -0.01] -> 2 wins / 4 = 0.5
    assert win_rate(returns) == pytest.approx(0.5)


def test_win_rate_all_zero_returns_zero():
    assert win_rate(np.zeros(10)) == 0.0


def test_summarize_returns_all_keys():
    equity = np.array([1000.0, 1010.0, 990.0, 1050.0, 1100.0])
    returns = np.diff(equity, prepend=equity[0]) / equity[0]
    summary = summarize(equity, returns)
    assert set(summary.keys()) == {
        "total_return",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
    }
    assert all(isinstance(v, float) for v in summary.values())
