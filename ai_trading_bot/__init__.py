"""ai_trading_bot -- backtest-only trading strategy research toolkit.

SAFETY NOTE: This project never connects to a live broker and never places
real orders. All data is either historical/mock OHLCV data or a synthetic
random-walk generator. See README.md for details. Nothing here is financial
advice.
"""

__all__ = ["data", "indicators", "strategy", "backtest", "metrics"]
