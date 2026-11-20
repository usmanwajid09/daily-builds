# finance_api

A small Flask personal finance API: income/expense transactions with
categories and dates, kept in memory (a fresh store per `create_app()`
call, nothing written to disk). It computes the balance, per-category
totals, a monthly summary, and budget alerts when spending approaches
or passes a category limit.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/transactions` | Record a transaction: `{"description", "amount", "type": "income"\|"expense", "category", "date": "YYYY-MM-DD"}` |
| GET | `/transactions` | List transactions; `?type=`, `?category=`, `?month=YYYY-MM` |
| GET | `/transactions/<id>` | Fetch one transaction |
| PUT | `/transactions/<id>` | Update description, amount, type, category and/or date |
| DELETE | `/transactions/<id>` | Delete a transaction |
| GET | `/balance` | Total income, total expense and balance |
| GET | `/categories` | Per-category income/expense totals and counts |
| GET | `/summary?month=YYYY-MM` | Monthly (or all-time) income, expense, balance and per-category totals |
| POST | `/budgets` | Set a budget: `{"category", "limit"}` |
| GET | `/budgets` | List budgets |
| PUT | `/budgets/<category>` | Change a budget's limit |
| DELETE | `/budgets/<category>` | Remove a budget |
| GET | `/budgets/alerts` | Per-budget spend vs limit with status `ok` / `warning` / `over` |

A transaction is `{"id", "description", "amount", "type", "category",
"date"}`; a budget is `{"category", "limit"}`. Amounts must be
positive and are stored rounded to two decimals. Budget status is
`warning` once spending reaches 80% of the limit and `over` once it
reaches or passes the limit.

Validation: missing/empty `description` or `category`, a
non-positive `amount`, a bad `type`, a date that is not `YYYY-MM-DD`,
or a bad `month` filter returns JSON `400`; an unknown transaction or
budget category returns JSON `404`.

## Run

```bash
python finance_api/demo.py
python -m pytest tests/test_finance_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/transactions -H 'Content-Type: application/json' \
  -d '{"description": "Groceries", "amount": 8000, "type": "expense",
       "category": "Food", "date": "2026-11-05"}'

curl localhost:5000/balance
curl 'localhost:5000/summary?month=2026-11'
curl -X POST localhost:5000/budgets -H 'Content-Type: application/json' \
  -d '{"category": "Food", "limit": 10000}'
curl localhost:5000/budgets/alerts
```
