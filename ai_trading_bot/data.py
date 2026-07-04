"""Historical OHLCV data ingestion.

Two sources are supported:

1. `load_csv(path)` -- ingest real historical daily-bar data from a CSV file
   with columns `date,open,high,low,close,volume` (case-insensitive, common
   aliases like `Adj Close` are handled). This is the path to use with real
   historical data the user already has on disk (e.g. exported from a
   broker's history page or a public dataset).

2. `generate_synthetic_ohlcv(...)` -- a deterministic (seeded) synthetic
   random-walk price generator used for backtesting/demo purposes when no
   real data file is supplied. It produces realistic-looking daily bars
   (open/high/low/close/volume) with configurable drift and volatility.

Nothing in this module contacts a broker, a brokerage API, or places any
order. It only ever reads/writes plain CSV files and/or generates numbers
in-process.
"""
from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OHLCVSeries:
    """A simple container for a daily OHLCV time series.

    All arrays are 1-D numpy arrays of equal length, ordered oldest -> newest.
    """

    dates: list  # list[datetime.date]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.dates)
        for name in ("open", "high", "low", "close", "volume"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"OHLCVSeries: '{name}' has length {len(arr)}, expected {n}"
                )
        if n == 0:
            raise ValueError("OHLCVSeries: series must not be empty")

    def __len__(self) -> int:
        return len(self.dates)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            for i in range(len(self)):
                writer.writerow(
                    [
                        self.dates[i].isoformat(),
                        self.open[i],
                        self.high[i],
                        self.low[i],
                        self.close[i],
                        self.volume[i],
                    ]
                )


# Column-name aliases we understand when reading a CSV, mapped to the
# canonical field name. Matching is case-insensitive and ignores spaces.
_COLUMN_ALIASES = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjclose": "close",
    "adjustedclose": "close",
    "volume": "volume",
    "vol": "volume",
}


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


def load_csv(path: str) -> OHLCVSeries:
    """Load a daily OHLCV series from a CSV file on disk.

    Expected columns (any case/order): date, open, high, low, close, volume.
    'Adj Close' / 'adjusted close' is accepted as an alias for close.
    Rows are sorted by date ascending regardless of file order.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row found")
        colmap = {}
        for raw in reader.fieldnames:
            key = _normalize_header(raw)
            if key in _COLUMN_ALIASES:
                colmap[_COLUMN_ALIASES[key]] = raw
        missing = {"date", "open", "high", "low", "close"} - set(colmap)
        if missing:
            raise ValueError(
                f"{path}: missing required column(s) {sorted(missing)}; "
                f"found headers {reader.fieldnames}"
            )
        has_volume = "volume" in colmap
        for line_no, row in enumerate(reader, start=2):
            try:
                date = _dt.date.fromisoformat(row[colmap["date"]].strip()[:10])
                o = float(row[colmap["open"]])
                h = float(row[colmap["high"]])
                lo = float(row[colmap["low"]])
                c = float(row[colmap["close"]])
                v = float(row[colmap["volume"]]) if has_volume else 0.0
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}: bad data on line {line_no}: {exc}") from exc
            rows.append((date, o, h, lo, c, v))

    if not rows:
        raise ValueError(f"{path}: file has a header but no data rows")

    rows.sort(key=lambda r: r[0])
    dates = [r[0] for r in rows]
    open_ = np.array([r[1] for r in rows], dtype=float)
    high = np.array([r[2] for r in rows], dtype=float)
    low = np.array([r[3] for r in rows], dtype=float)
    close = np.array([r[4] for r in rows], dtype=float)
    volume = np.array([r[5] for r in rows], dtype=float)
    return OHLCVSeries(dates, open_, high, low, close, volume)


def generate_synthetic_ohlcv(
    n_days: int = 500,
    start_price: float = 100.0,
    annual_drift: float = 0.06,
    annual_vol: float = 0.25,
    start_date: _dt.date | None = None,
    seed: int = 42,
) -> OHLCVSeries:
    """Generate a deterministic synthetic daily OHLCV series.

    Uses a geometric-Brownian-motion-style random walk for the close price
    (so returns are stationary and roughly log-normal, similar to a real
    equity), then derives open/high/low around each day's close/prev-close
    with a small synthetic intraday range, and a lightly randomized volume.

    This is NOT real market data. It exists so the rest of the pipeline
    (indicators, strategy, backtest, metrics) can be developed and tested
    without any network access or brokerage connection.
    """
    if n_days < 2:
        raise ValueError("n_days must be >= 2")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = _dt.date(2023, 1, 2)

    trading_days_per_year = 252
    mu_daily = annual_drift / trading_days_per_year
    sigma_daily = annual_vol / np.sqrt(trading_days_per_year)

    # Log-return random walk -> close prices.
    log_returns = rng.normal(loc=mu_daily - 0.5 * sigma_daily**2, scale=sigma_daily, size=n_days)
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    close = np.exp(log_prices)

    prev_close = np.empty(n_days)
    prev_close[0] = start_price
    prev_close[1:] = close[:-1]

    # Open gaps slightly from the prior close.
    gap = rng.normal(loc=0.0, scale=sigma_daily * 0.5, size=n_days)
    open_ = prev_close * np.exp(gap)

    # Intraday range around max(open, close)/min(open, close).
    intraday_extra = np.abs(rng.normal(loc=sigma_daily, scale=sigma_daily * 0.5, size=n_days))
    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    high = hi_base * (1.0 + intraday_extra)
    low = lo_base * (1.0 - intraday_extra)
    low = np.minimum(low, np.minimum(open_, close) * 0.999)  # guard against rounding

    # Volume: baseline with noise, mildly elevated on big absolute moves.
    base_volume = 1_000_000
    move_size = np.abs(log_returns) / sigma_daily
    volume = base_volume * (1.0 + 0.3 * move_size) * rng.uniform(0.7, 1.3, size=n_days)
    volume = volume.round().astype(float)

    dates = []
    d = start_date
    added = 0
    while added < n_days:
        if d.weekday() < 5:  # skip weekends to look like real trading days
            dates.append(d)
            added += 1
        d = d + _dt.timedelta(days=1)

    return OHLCVSeries(dates, open_, high, low, close, volume)
