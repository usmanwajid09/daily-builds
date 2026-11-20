"""Flask mock crypto price API.

All prices are fixed, seeded values generated at import time — no
network calls, no live data, so every response is deterministic.
Each coin has 31 daily closing prices (30 days ago through today);
the current price is the coin's listed base price. Endpoints:

- ``GET /coins`` — all coins: symbol, name, price, 24h change, market cap
- ``GET /coins/<symbol>`` — one coin, with its 7-day change
- ``GET /coins/<symbol>/history?days=7`` — daily prices (``days`` 1–30)
- ``GET /coins/<symbol>/change?days=7`` — % change over the window
- ``GET /convert?from=BTC&to=ETH&amount=1`` — convert between coins
- ``GET /top?limit=3`` — top coins by market cap

Symbols are case-insensitive in the URL. Prices are in USD.
"""
import random
from datetime import date, timedelta

from flask import Flask, jsonify, request

# symbol -> (name, base price in USD, circulating supply)
_COINS = {
    "BTC": ("Bitcoin", 97432.10, 19_800_000),
    "ETH": ("Ethereum", 3412.55, 120_400_000),
    "SOL": ("Solana", 238.40, 479_000_000),
    "DOGE": ("Dogecoin", 0.4123, 147_700_000_000),
    "ADA": ("Cardano", 0.9812, 35_600_000_000),
    "XRP": ("XRP", 2.4510, 57_100_000_000),
    "LTC": ("Litecoin", 112.75, 75_300_000),
    "LINK": ("Chainlink", 22.84, 657_000_000),
}

_HISTORY_DAYS = 30  # prices stored for day -30 .. day 0 (today)


def _build_history(symbol, base_price):
    """Deterministic 31-day price walk ending exactly at ``base_price``."""
    seed = sum(ord(ch) for ch in symbol) * 1000 + 42
    rng = random.Random(seed)
    prices = [base_price * 0.97]
    for _ in range(_HISTORY_DAYS):
        prices.append(prices[-1] * (1 + rng.uniform(-0.04, 0.04)))
    scale = base_price / prices[-1]
    return [round(p * scale, 4) for p in prices]


def _pct_change(new, old):
    return round((new - old) / old * 100, 2)


# symbol -> {"name": str, "price_usd": float, "market_cap_usd": float,
#             "history": [float, ...]}  (history[0] is 30 days ago)
COINS = {}
for _symbol, (_name, _base, _supply) in _COINS.items():
    _history = _build_history(_symbol, _base)
    COINS[_symbol] = {
        "name": _name,
        "price_usd": _base,
        "market_cap_usd": round(_base * _supply, 2),
        "history": _history,
    }


def _summary(symbol, coin):
    return {
        "symbol": symbol,
        "name": coin["name"],
        "price_usd": coin["price_usd"],
        "change_24h_pct": _pct_change(coin["history"][-1], coin["history"][-2]),
        "market_cap_usd": coin["market_cap_usd"],
    }


def create_app():
    """Return a Flask app serving the seeded mock crypto data."""
    app = Flask(__name__)

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_coin(symbol):
        coin = COINS.get(symbol.upper())
        if coin is None:
            return None, _error(f"unknown coin '{symbol}'", 404)
        return coin, None

    @app.get("/coins")
    def list_coins():
        coins = [_summary(s, c) for s, c in sorted(COINS.items())]
        return jsonify(coins)

    @app.get("/coins/<symbol>")
    def get_coin(symbol):
        coin, err = _get_coin(symbol)
        if err is not None:
            return err
        detail = _summary(symbol.upper(), coin)
        detail["change_7d_pct"] = _pct_change(
            coin["history"][-1], coin["history"][-8]
        )
        return jsonify(detail)

    @app.get("/coins/<symbol>/history")
    def coin_history(symbol):
        coin, err = _get_coin(symbol)
        if err is not None:
            return err
        days_raw = request.args.get("days", "7")
        try:
            days = int(days_raw)
        except ValueError:
            return _error("days must be an integer between 1 and 30", 400)
        if isinstance(days, bool) or not 1 <= days <= _HISTORY_DAYS:
            return _error("days must be an integer between 1 and 30", 400)
        today = date.today()
        history = coin["history"][-days:]
        rows = [
            {
                "date": (today - timedelta(days=days - 1 - i)).isoformat(),
                "price_usd": price,
            }
            for i, price in enumerate(history)
        ]
        return jsonify({"symbol": symbol.upper(), "days": days, "prices": rows})

    @app.get("/coins/<symbol>/change")
    def coin_change(symbol):
        coin, err = _get_coin(symbol)
        if err is not None:
            return err
        days_raw = request.args.get("days", "7")
        try:
            days = int(days_raw)
        except ValueError:
            return _error("days must be an integer between 1 and 30", 400)
        if not 1 <= days <= _HISTORY_DAYS:
            return _error("days must be an integer between 1 and 30", 400)
        history = coin["history"]
        start = history[-1 - days]
        end = history[-1]
        return jsonify(
            {
                "symbol": symbol.upper(),
                "days": days,
                "start_price_usd": start,
                "end_price_usd": end,
                "change_pct": _pct_change(end, start),
            }
        )

    @app.get("/convert")
    def convert():
        from_sym = request.args.get("from", "").upper()
        to_sym = request.args.get("to", "").upper()
        if from_sym not in COINS:
            return _error(f"unknown coin '{request.args.get('from', '')}'", 400)
        if to_sym not in COINS:
            return _error(f"unknown coin '{request.args.get('to', '')}'", 400)
        amount_raw = request.args.get("amount", "1")
        try:
            amount = float(amount_raw)
        except ValueError:
            return _error("amount must be a number", 400)
        if amount <= 0:
            return _error("amount must be a positive number", 400)
        rate = COINS[from_sym]["price_usd"] / COINS[to_sym]["price_usd"]
        return jsonify(
            {
                "from": from_sym,
                "to": to_sym,
                "amount": amount,
                "rate": round(rate, 6),
                "converted": round(amount * rate, 6),
            }
        )

    @app.get("/top")
    def top_coins():
        limit_raw = request.args.get("limit", "5")
        try:
            limit = int(limit_raw)
        except ValueError:
            return _error("limit must be a positive integer", 400)
        if not 1 <= limit <= len(COINS):
            return _error(f"limit must be between 1 and {len(COINS)}", 400)
        ranked = sorted(
            COINS.items(), key=lambda kv: kv[1]["market_cap_usd"], reverse=True
        )
        return jsonify([_summary(s, c) for s, c in ranked[:limit]])

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
