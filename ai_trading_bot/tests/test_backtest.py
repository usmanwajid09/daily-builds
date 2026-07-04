import numpy as np
import pytest

from ai_trading_bot.backtest import buy_and_hold_baseline, run_backtest


def test_run_backtest_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        run_backtest(np.array([1.0, 2.0, 3.0]), np.array([1, 0]))


def test_run_backtest_rejects_too_short_series():
    with pytest.raises(ValueError):
        run_backtest(np.array([1.0]), np.array([1]))


def test_run_backtest_all_flat_preserves_capital():
    close = np.array([100.0, 105.0, 95.0, 110.0])
    signal = np.zeros(4, dtype=int)
    result = run_backtest(close, signal, initial_capital=1000.0, transaction_cost_bps=0.0)
    np.testing.assert_allclose(result.equity_curve, 1000.0)
    assert result.trades == 0
    assert result.final_capital == 1000.0


def test_run_backtest_no_lookahead_bar_zero_is_always_flat():
    # Bar 0 has no prior signal to act on, so its effective position is
    # always 0 regardless of signal[0] -- there is no bar "-1 -> 0" return
    # to decide about, so equity can't move on bar 0 itself.
    close = np.array([100.0, 200.0, 200.0])
    signal = np.array([1, 1, 1])
    result = run_backtest(close, signal, initial_capital=1000.0, transaction_cost_bps=0.0)
    assert result.positions[0] == 0
    assert result.equity_curve[0] == pytest.approx(1000.0)
    # signal[0]=1 (decided using bar 0's close) IS acted on for the bar
    # 0 -> 1 return, since that decision was available before that return
    # happened -- this is the earliest any no-lookahead backtest can react.
    assert result.equity_curve[1] == pytest.approx(2000.0)
    # bar 1 -> 2 return is 0%, so no further change expected.
    assert result.equity_curve[2] == pytest.approx(2000.0)


def test_run_backtest_matches_hand_computed_long_only():
    close = np.array([100.0, 110.0, 121.0])  # +10% each bar
    signal = np.array([1, 1, 1])
    result = run_backtest(close, signal, initial_capital=1000.0, transaction_cost_bps=0.0)
    # effective_position[1]=signal[0]=1 captures bar0->1 (+10%);
    # effective_position[2]=signal[1]=1 captures bar1->2 (+10%).
    expected = 1000.0 * 1.10 * 1.10
    assert result.equity_curve[-1] == pytest.approx(expected)


def test_run_backtest_transaction_cost_reduces_equity():
    close = np.array([100.0, 100.0, 100.0, 100.0])
    signal = np.array([1, 0, 1, 0])  # flips every bar -> trades every bar
    no_cost = run_backtest(close, signal, initial_capital=1000.0, transaction_cost_bps=0.0)
    with_cost = run_backtest(close, signal, initial_capital=1000.0, transaction_cost_bps=50.0)
    assert with_cost.final_capital < no_cost.final_capital
    assert with_cost.trades == no_cost.trades
    assert with_cost.trades > 0


def test_run_backtest_rejects_bad_params():
    close = np.array([100.0, 101.0, 102.0])
    signal = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        run_backtest(close, signal, initial_capital=0)
    with pytest.raises(ValueError):
        run_backtest(close, signal, transaction_cost_bps=-1)


def test_buy_and_hold_baseline_tracks_close_after_warmup():
    close = np.array([100.0, 110.0, 121.0, 133.1])
    result = buy_and_hold_baseline(close, initial_capital=1000.0)
    # Position is 1 for every bar from bar 1 onward (bar 0 is always flat,
    # see no-lookahead note above), so equity tracks price growth measured
    # from close[0] -- the earliest point a no-lookahead backtest can act.
    expected_from_bar1 = 1000.0 * close[1:] / close[0]
    np.testing.assert_allclose(result.equity_curve[1:], expected_from_bar1)
