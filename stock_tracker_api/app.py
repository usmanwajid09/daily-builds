"""Flask mock stock tracker API.

Prices are seeded mock data (fixed at app start, nothing is fetched
from the network and nothing is written to disk). All responses are
JSON. Endpoints:

- ``GET /stocks`` — all seeded tickers with their current mock price
- ``GET /stocks/<ticker>`` — one ticker's current price
- ``GET /stocks/<ticker>/history`` — 7 days of mock closing prices
- ``POST /watchlist`` — add a ticker (``{"ticker": str}``)
- ``GET /watchlist`` — tickers on the watchlist with prices
- ``DELETE /watchlist/<ticker>`` — remove a ticker
- ``POST /portfolio/buy`` — buy shares (``{"ticker": str, "shares": int}``)
- ``POST /portfolio/sell`` — sell shares (``{"ticker": str, "shares": int}``)
- ``GET /portfolio`` — holdings valued at current prices plus total value

Tickers are case-insensitive. An unknown ticker returns ``404``;
adding a ticker already on the watchlist returns ``409``; selling
more shares than held (or a non-positive share count) returns ``400``.
"""
from flask import Flask, jsonify, request

# Seeded mock prices (USD per share), fixed for the life of the app.
SEEDED_PRICES = {
    "AAPL": 175.50,
    "MSFT": 420.25,
    "GOOGL": 178.30,
    "TSLA": 245.80,
    "NVDA": 135.60,
}

# Deterministic mock history: 7 days ending 2026-11-26, each day's
# close expressed as a fixed fraction of the current price.
HISTORY_DATES = [
    "2026-11-20",
    "2026-11-21",
    "2026-11-22",
    "2026-11-23",
    "2026-11-24",
    "2026-11-25",
    "2026-11-26",
]
HISTORY_FACTORS = [0.970, 0.985, 1.000, 0.990, 1.010, 1.005, 1.000]


def create_app():
    """Return a Flask app with seeded mock prices and an empty portfolio."""
    app = Flask(__name__)
    app.config["WATCHLIST"] = set()
    app.config["PORTFOLIO"] = {}

    def _error(message, status):
        return jsonify({"error": message}), status

    def _ticker(value):
        """Normalise a ticker string, or return None if not a string."""
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip().upper()

    def _known(ticker):
        if ticker not in SEEDED_PRICES:
            return _error(f"unknown ticker {ticker}", 404)
        return None

    def _history(ticker):
        base = SEEDED_PRICES[ticker]
        return [
            {"date": day, "close": round(base * factor, 2)}
            for day, factor in zip(HISTORY_DATES, HISTORY_FACTORS)
        ]

    def _validate_trade(data):
        """Return (ticker, shares, error) for a buy/sell request body."""
        if not isinstance(data, dict):
            return None, None, _error("request body must be a JSON object", 400)
        ticker = _ticker(data.get("ticker"))
        if ticker is None:
            return None, None, _error("ticker is required", 400)
        err = _known(ticker)
        if err is not None:
            return None, None, err
        shares = data.get("shares")
        if isinstance(shares, bool) or not isinstance(shares, int) or shares <= 0:
            return None, None, _error("shares must be a positive integer", 400)
        return ticker, shares, None

    @app.get("/stocks")
    def list_stocks():
        return jsonify(
            [
                {"ticker": ticker, "price": price}
                for ticker, price in sorted(SEEDED_PRICES.items())
            ]
        )

    @app.get("/stocks/<ticker>")
    def get_stock(ticker):
        ticker = ticker.upper()
        err = _known(ticker)
        if err is not None:
            return err
        return jsonify({"ticker": ticker, "price": SEEDED_PRICES[ticker]})

    @app.get("/stocks/<ticker>/history")
    def stock_history(ticker):
        ticker = ticker.upper()
        err = _known(ticker)
        if err is not None:
            return err
        return jsonify({"ticker": ticker, "history": _history(ticker)})

    @app.post("/watchlist")
    def add_to_watchlist():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        ticker = _ticker(data.get("ticker"))
        if ticker is None:
            return _error("ticker is required", 400)
        err = _known(ticker)
        if err is not None:
            return err
        if ticker in app.config["WATCHLIST"]:
            return _error(f"{ticker} is already on the watchlist", 409)
        app.config["WATCHLIST"].add(ticker)
        return (
            jsonify({"ticker": ticker, "price": SEEDED_PRICES[ticker]}),
            201,
        )

    @app.get("/watchlist")
    def list_watchlist():
        return jsonify(
            [
                {"ticker": ticker, "price": SEEDED_PRICES[ticker]}
                for ticker in sorted(app.config["WATCHLIST"])
            ]
        )

    @app.delete("/watchlist/<ticker>")
    def remove_from_watchlist(ticker):
        ticker = ticker.upper()
        if ticker not in app.config["WATCHLIST"]:
            return _error(f"{ticker} is not on the watchlist", 404)
        app.config["WATCHLIST"].discard(ticker)
        return jsonify({"removed": ticker})

    @app.post("/portfolio/buy")
    def buy_shares():
        ticker, shares, err = _validate_trade(request.get_json(silent=True))
        if err is not None:
            return err
        portfolio = app.config["PORTFOLIO"]
        portfolio[ticker] = portfolio.get(ticker, 0) + shares
        return jsonify(
            {
                "ticker": ticker,
                "shares": portfolio[ticker],
                "price": SEEDED_PRICES[ticker],
                "value": round(portfolio[ticker] * SEEDED_PRICES[ticker], 2),
            }
        )

    @app.post("/portfolio/sell")
    def sell_shares():
        ticker, shares, err = _validate_trade(request.get_json(silent=True))
        if err is not None:
            return err
        portfolio = app.config["PORTFOLIO"]
        held = portfolio.get(ticker, 0)
        if shares > held:
            return _error(
                f"cannot sell {shares} shares of {ticker}: only {held} held", 400
            )
        remaining = held - shares
        if remaining == 0:
            del portfolio[ticker]
        else:
            portfolio[ticker] = remaining
        return jsonify(
            {
                "ticker": ticker,
                "shares": remaining,
                "price": SEEDED_PRICES[ticker],
                "value": round(remaining * SEEDED_PRICES[ticker], 2),
            }
        )

    @app.get("/portfolio")
    def get_portfolio():
        portfolio = app.config["PORTFOLIO"]
        holdings = []
        total = 0.0
        for ticker in sorted(portfolio):
            value = round(portfolio[ticker] * SEEDED_PRICES[ticker], 2)
            total += value
            holdings.append(
                {
                    "ticker": ticker,
                    "shares": portfolio[ticker],
                    "price": SEEDED_PRICES[ticker],
                    "value": value,
                }
            )
        return jsonify({"holdings": holdings, "total_value": round(total, 2)})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
