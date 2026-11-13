# Expense Tracker

A tiny expense-tracker API built with Flask. Expenses live in an
in-memory store (fresh on every app start), with validation on every
field and totals computed per category and per month.

## Why

Tracking spending is mostly adding validated records and summing them
by category and by month — the same shape as a surprising number of
real backends, minus the database.

## Usage

```bash
pip install -r requirements.txt
python -m expense_tracker.demo
```

The demo adds three expenses and prints the per-category totals, the
grand total, and the November 2026 total. To run the app for real:

```bash
flask --app "expense_tracker.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/expenses` | Add one: `{"amount", "category", "date", "description"?}` |
| GET | `/expenses` | List all (`?category=food` filters) |
| GET | `/expenses/<id>` | Fetch one expense |
| DELETE | `/expenses/<id>` | Delete one expense |
| GET | `/totals` | `{"total", "by_category", "count"}` |
| GET | `/totals/<YYYY-MM>` | `{"month", "total", "count"}` |

Validation: `amount` must be a number greater than 0, `category` a
non-empty string, `date` a real calendar date in `YYYY-MM-DD` format.
Bad input returns `400` with `{"error": "..."}`; unknown ids return
`404`.

## Run tests

```bash
python -m pytest tests/test_expense_tracker.py -v
```
