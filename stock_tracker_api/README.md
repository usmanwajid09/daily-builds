# Stock Tracker API

A tiny REST mock stock tracker built with Flask. Prices are seeded
mock data (AAPL, MSFT, GOOGL, TSLA, NVDA) — fixed at app start,
nothing is fetched from the network and nothing is written to disk.
Track a watchlist, buy and sell shares, and see the portfolio valued
at current prices, plus a 7-day price history per ticker.

## Why

A portfolio is a small ledger valued against a price table: holdings
times current prices. This app shows that shape — a seeded price
list, a watchlist, buy/sell with the obvious guard (you cannot sell
more than you hold), and a total value computed live from prices.

## Usage

```bash
pip install -r requirements.txt
python -m stock_tracker_api.demo
```

The demo lists the seeded prices, adds two tickers to the watchlist,
buys 10 AAPL and 5 NVDA, and prints the portfolio value ($2,433.00).

To run the app for real:

```bash
flask --app "stock_tracker_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/stocks` | All seeded tickers with current mock prices |
| GET | `/stocks/<ticker>` | One ticker's current price |
| GET | `/stocks/<ticker>/history` | 7 days of mock closing prices |
| POST | `/watchlist` | Add a ticker: `{"ticker": str}` |
| GET | `/watchlist` | Tickers on the watchlist with prices |
| DELETE | `/watchlist/<ticker>` | Remove a ticker |
| POST | `/portfolio/buy` | Buy shares: `{"ticker": str, "shares": int}` |
| POST | `/portfolio/sell` | Sell shares: `{"ticker": str, "shares": int}` |
| GET | `/portfolio` | Holdings valued at current prices plus total value |

Tickers are case-insensitive. A holding looks like
`{"ticker": "AAPL", "shares": 10, "price": 175.5, "value": 1755.0}`
and the portfolio looks like
`{"holdings": [...], "total_value": 2433.0}`.

Validation errors return `400` with `{"error": "..."}`; unknown
tickers and routes return `404`. Adding a ticker already on the
watchlist returns `409`; selling more shares than held (or a
non-positive share count) returns `400`.

## Demo output

```
Seeded prices:
  AAPL: $175.50
  GOOGL: $178.30
  MSFT: $420.25
  NVDA: $135.60
  TSLA: $245.80
Watchlist: added AAPL at $175.50
Watchlist: added NVDA at $135.60
Bought 10 AAPL at $175.50 -> 10 shares worth $1755.00
Portfolio:
  AAPL: 10 shares x $175.50 = $1755.00
  NVDA: 5 shares x $135.60 = $678.00
Total portfolio value: $2433.00
AAPL history: 7 days, latest close $175.50
```

## Run tests

```bash
python -m pytest tests/test_stock_tracker_api.py -v
```
