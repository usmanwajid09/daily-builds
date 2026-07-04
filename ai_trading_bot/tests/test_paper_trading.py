import datetime as dt

import numpy as np
import pytest

from ai_trading_bot.paper_trading import PaperAccount, run_mock_paper_trading


def test_paper_account_rejects_negative_cash():
    with pytest.raises(ValueError):
        PaperAccount(cash=-1.0)


def test_paper_account_go_long_then_flat_roundtrip_no_cost():
    acct = PaperAccount(cash=1000.0)
    acct.go_long(dt.date(2023, 1, 1), price=100.0, transaction_cost_bps=0.0)
    assert acct.cash == 0.0
    assert acct.shares == pytest.approx(10.0)
    assert acct.is_long()
    acct.go_flat(dt.date(2023, 1, 2), price=110.0, transaction_cost_bps=0.0)
    assert acct.shares == 0.0
    assert acct.cash == pytest.approx(1100.0)
    assert not acct.is_long()
    assert len(acct.trade_log) == 2
    assert acct.trade_log[0]["action"] == "BUY"
    assert acct.trade_log[1]["action"] == "SELL"


def test_paper_account_go_long_is_idempotent():
    acct = PaperAccount(cash=1000.0)
    acct.go_long(dt.date(2023, 1, 1), price=100.0, transaction_cost_bps=0.0)
    acct.go_long(dt.date(2023, 1, 2), price=50.0, transaction_cost_bps=0.0)  # already long, no-op
    assert len(acct.trade_log) == 1
    assert acct.shares == pytest.approx(10.0)  # unchanged from the first buy


def test_paper_account_go_flat_is_idempotent():
    acct = PaperAccount(cash=1000.0)
    acct.go_flat(dt.date(2023, 1, 1), price=100.0, transaction_cost_bps=0.0)  # never bought
    assert len(acct.trade_log) == 0
    assert acct.cash == 1000.0


def test_paper_account_rejects_nonpositive_trade_price():
    acct = PaperAccount(cash=1000.0)
    with pytest.raises(ValueError):
        acct.go_long(dt.date(2023, 1, 1), price=0.0, transaction_cost_bps=0.0)


def test_paper_account_transaction_cost_reduces_proceeds():
    acct_free = PaperAccount(cash=1000.0)
    acct_free.go_long(dt.date(2023, 1, 1), price=100.0, transaction_cost_bps=0.0)
    acct_free.go_flat(dt.date(2023, 1, 2), price=100.0, transaction_cost_bps=0.0)

    acct_cost = PaperAccount(cash=1000.0)
    acct_cost.go_long(dt.date(2023, 1, 1), price=100.0, transaction_cost_bps=50.0)
    acct_cost.go_flat(dt.date(2023, 1, 2), price=100.0, transaction_cost_bps=50.0)

    assert acct_cost.cash < acct_free.cash


def test_run_mock_paper_trading_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        run_mock_paper_trading(
            dates=[dt.date(2023, 1, 1), dt.date(2023, 1, 2)],
            close=np.array([100.0, 101.0, 102.0]),
            desired_position=np.array([1, 1]),
        )


def test_run_mock_paper_trading_rejects_bad_params():
    dates = [dt.date(2023, 1, 1), dt.date(2023, 1, 2)]
    close = np.array([100.0, 101.0])
    signal = np.array([1, 1])
    with pytest.raises(ValueError):
        run_mock_paper_trading(dates, close, signal, initial_capital=0)
    with pytest.raises(ValueError):
        run_mock_paper_trading(dates, close, signal, transaction_cost_bps=-1)


def test_run_mock_paper_trading_no_lookahead_bar_zero():
    dates = [dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 1, 3)]
    close = np.array([100.0, 200.0, 200.0])
    signal = np.array([1, 1, 1])
    account, equity_curve = run_mock_paper_trading(
        dates, close, signal, initial_capital=1000.0, transaction_cost_bps=0.0
    )
    # bar 0 is observation only (nothing decided yet) -- same no-lookahead
    # starting point as backtest.run_backtest.
    assert equity_curve[0] == pytest.approx(1000.0)
    # signal[0]=1 (decided at bar 0's close) is actable at bar 1 -- but
    # UNLIKE run_backtest's idealized "mark-to-market from the prior
    # close" model, this loop actually buys AT bar 1's price (200), since
    # that's the earliest real price a live order could fill at. So the
    # 100->200 move is NOT captured -- equity is unchanged right after the
    # buy (now in shares instead of cash), matching a realistic "you can't
    # buy at yesterday's price" fill assumption.
    assert equity_curve[1] == pytest.approx(1000.0)
    # already long at bar 2 (signal[1]=1, no-op) -- price unchanged, so
    # equity is unchanged too.
    assert equity_curve[2] == pytest.approx(1000.0)


def test_run_mock_paper_trading_matches_manual_account_trace():
    dates = [dt.date(2023, 1, i + 1) for i in range(4)]
    close = np.array([100.0, 110.0, 105.0, 120.0])
    signal = np.array([1, 1, 0, 1])  # long, long, flat, long
    account, equity_curve = run_mock_paper_trading(
        dates, close, signal, initial_capital=1000.0, transaction_cost_bps=0.0
    )
    # bar1: act on signal[0]=1 -> buy at 110 -> shares = 1000/110
    shares_bought = 1000.0 / 110.0
    assert equity_curve[1] == pytest.approx(shares_bought * 110.0)
    # bar2: act on signal[1]=1 -> already long, no-op -> mark to market at 105
    assert equity_curve[2] == pytest.approx(shares_bought * 105.0)
    # bar3: act on signal[2]=0 -> sell at 120
    proceeds = shares_bought * 120.0
    assert equity_curve[3] == pytest.approx(proceeds)
    assert account.cash == pytest.approx(proceeds)
    assert not account.is_long()
    assert len(account.trade_log) == 2  # one BUY, one SELL
