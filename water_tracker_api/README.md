# Water Tracker API

A tiny REST API for daily water intake built with Flask. Intake logs
live in an in-memory store (fresh on every app start — nothing is
written to disk).

## Why

Drinking enough water is a habit, and habits need a visible streak.
This app logs glasses or millilitres, compares each day against a
daily target, and counts consecutive days that met the target.

## Usage

```bash
pip install -r requirements.txt
python water_tracker_api/demo.py
```

## Endpoints

- `GET /target` — current daily target in ml (default 2000)
- `PUT /target` — set the daily target (`{"target_ml": n}`)
- `POST /intake` — log water (`{"amount_ml": n}` or `{"glasses": n}`,
  optional `"date"`, 1 glass = 250 ml)
- `GET /intake?date=YYYY-MM-DD` — entries and total for a day
- `GET /summary/<date>` — total vs target for one day
- `GET /streak` — consecutive days meeting the target
- `GET /weekly?date=YYYY-MM-DD` — 7-day summary ending on `date`
```

## Tests

```bash
python -m pytest tests/test_water_tracker_api.py -q
```
