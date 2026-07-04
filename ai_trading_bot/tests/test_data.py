import datetime as dt
import os
import tempfile

import numpy as np
import pytest

from ai_trading_bot.data import OHLCVSeries, generate_synthetic_ohlcv, load_csv


def test_generate_synthetic_ohlcv_shapes_and_determinism():
    s1 = generate_synthetic_ohlcv(n_days=100, seed=7)
    s2 = generate_synthetic_ohlcv(n_days=100, seed=7)
    assert len(s1) == 100
    assert np.array_equal(s1.close, s2.close), "same seed must reproduce same series"
    # different seed -> different data
    s3 = generate_synthetic_ohlcv(n_days=100, seed=8)
    assert not np.array_equal(s1.close, s3.close)


def test_generate_synthetic_ohlcv_ohlc_consistency():
    s = generate_synthetic_ohlcv(n_days=250, seed=1)
    assert np.all(s.high >= s.open)
    assert np.all(s.high >= s.close)
    assert np.all(s.low <= s.open)
    assert np.all(s.low <= s.close)
    assert np.all(s.close > 0)
    assert np.all(s.volume >= 0)


def test_generate_synthetic_ohlcv_skips_weekends():
    s = generate_synthetic_ohlcv(n_days=30, seed=1, start_date=dt.date(2023, 1, 2))
    weekdays = {d.weekday() for d in s.dates}
    assert weekdays.issubset({0, 1, 2, 3, 4})


def test_generate_synthetic_ohlcv_rejects_bad_input():
    with pytest.raises(ValueError):
        generate_synthetic_ohlcv(n_days=1)
    with pytest.raises(ValueError):
        generate_synthetic_ohlcv(n_days=10, start_price=-5)


def test_ohlcv_series_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        OHLCVSeries(
            dates=[dt.date(2023, 1, 2), dt.date(2023, 1, 3)],
            open=np.array([1.0]),
            high=np.array([1.0, 2.0]),
            low=np.array([1.0, 2.0]),
            close=np.array([1.0, 2.0]),
            volume=np.array([1.0, 2.0]),
        )


def test_ohlcv_series_rejects_empty():
    with pytest.raises(ValueError):
        OHLCVSeries([], np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))


def test_csv_roundtrip(tmp_path):
    s = generate_synthetic_ohlcv(n_days=20, seed=3)
    path = tmp_path / "series.csv"
    s.to_csv(str(path))
    loaded = load_csv(str(path))
    assert len(loaded) == len(s)
    assert loaded.dates == s.dates
    np.testing.assert_allclose(loaded.close, s.close, rtol=1e-9)


def test_load_csv_accepts_adj_close_alias(tmp_path):
    path = tmp_path / "aliased.csv"
    path.write_text(
        "Date,Open,High,Low,Adj Close,Volume\n"
        "2023-01-03,10,11,9,10.5,1000\n"
        "2023-01-02,9,10,8,9.5,900\n"
    )
    s = load_csv(str(path))
    # sorted ascending by date even though file had them descending
    assert s.dates == [dt.date(2023, 1, 2), dt.date(2023, 1, 3)]
    np.testing.assert_allclose(s.close, [9.5, 10.5])


def test_load_csv_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,open,high,close\n2023-01-02,1,2,1.5\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_csv(str(path))


def test_load_csv_bad_row_raises(tmp_path):
    path = tmp_path / "bad_row.csv"
    path.write_text(
        "date,open,high,low,close,volume\n"
        "2023-01-02,1,2,0.5,1.5,100\n"
        "not-a-date,1,2,0.5,1.5,100\n"
    )
    with pytest.raises(ValueError, match="bad data on line"):
        load_csv(str(path))


def test_load_csv_no_rows_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("date,open,high,low,close,volume\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_csv(str(path))
