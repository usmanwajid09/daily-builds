# crypto_price_api

A small Flask mock crypto price API: fixed, seeded prices for 8 coins,
kept in memory (nothing written to disk, no network calls). Each coin
has a deterministic 31-day price history generated from a fixed seed,
ending at its listed base price, so every response is identical on
every run.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| GET | `/coins` | All coins: symbol, name, price, 24h change, market cap |
| GET | `/coins/<symbol>` | One coin, with 7-day change |
| GET | `/coins/<symbol>/history?days=7` | Daily prices; `days` 1–30 |
| GET | `/coins/<symbol>/change?days=7` | % change over the window |
| GET | `/convert?from=BTC&to=ETH&amount=1` | Convert between coins |
| GET | `/top?limit=5` | Top coins by market cap |

Coins: BTC, ETH, SOL, DOGE, ADA, XRP, LTC, LINK. Symbols are
case-insensitive. Prices are in USD. A coin entry is `{"symbol",
"name", "price_usd", "change_24h_pct", "market_cap_usd"}`.

Validation: an unknown coin returns JSON `404`; `days` outside 1–30,
a `limit` outside 1–8, a bad `amount`, or an unknown coin in
`/convert` returns JSON `400`.

## Run

```bash
python crypto_price_api/demo.py
python -m pytest tests/test_crypto_price_api.py -v
```

## Examples

```bash
curl localhost:5000/coins
curl localhost:5000/coins/BTC/history?days=7
curl localhost:5000/coins/ETH/change?days=7
curl 'localhost:5000/convert?from=BTC&to=ETH&amount=0.5'
curl localhost:5000/top?limit=3
```
