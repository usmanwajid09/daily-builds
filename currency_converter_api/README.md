# Currency Converter API

A tiny Flask API that converts amounts between currencies using a
small built-in rate table. The rates are **demo rates** (units of each
currency per 1 USD) — they are not live market rates, and the API
labels every conversion result `"demo_rates": true`. Rates can be
added or updated at runtime; everything lives in memory and resets on
restart.

## Why

Currency conversion is a classic API shape: a lookup table, a
calculation, and a few query parameters. This app shows query-param
validation, derived values (the cross rate), and a mutable in-memory
table with proper create/update status codes.

## Usage

```bash
pip install -r requirements.txt
python -m currency_converter_api.demo
```

The demo lists the currencies, converts 10 USD to PKR, adds CHF, and
shows an unknown-currency error.

To run the app for real:

```bash
flask --app "currency_converter_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/currencies` | List all currencies and rates |
| GET | `/rates/<code>` | One currency's rate (404 if unknown) |
| GET | `/convert?from=USD&to=PKR&amount=10` | Convert; returns `amount`, `rate`, `result`, `demo_rates: true` |
| POST | `/rates` | Add/update a rate: `{"code": "EUR", "rate": 0.92}` — 201 for new, 200 for update |

Currency codes are 3 letters and case-insensitive (`usd` works).
Validation errors return `400` with `{"error": "..."}`; unknown
currencies on `/rates/<code>` and unknown routes return `404`.

## Run tests

```bash
python -m pytest tests/test_currency_converter_api.py -v
```
