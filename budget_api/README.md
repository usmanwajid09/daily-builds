# Budget API

A tiny monthly-budget tracker API built with Flask. Category budgets
and expenses live in an in-memory store (fresh on every app start —
nothing is written to disk), so it is ideal for learning REST
patterns and for testing with Flask's test client.

## Usage

```bash
pip install -r requirements.txt
python -m budget_api.demo
```

The demo sets a food budget of 20000 and a transport budget of
5000, adds three expenses, and prints the remaining amount and
over-budget flag for each category plus a totals summary.

To run the app for real:

```bash
flask --app "budget_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/budgets` | List categories with spent, remaining, over-budget flag |
| POST | `/budgets` | Set a budget: `{"category": str, "amount": number}` |
| GET | `/budgets/<category>` | One category's budget summary |
| DELETE | `/budgets/<category>` | Remove a budget and its expenses |
| POST | `/expenses` | Add an expense: `{"category": str, "amount": number, "note": str?}` |
| GET | `/expenses` | List expenses (optional `?category=` filter) |
| GET | `/summary` | Totals: budget, spent, remaining, over-budget categories |

Validation errors return `400` with `{"error": "..."}`; unknown
categories return `404`. Expenses can only be added to a category
that already has a budget. A category summary looks like
`{"category": "food", "budget": 20000, "spent": 5700,
"remaining": 14300, "over_budget": false}`.

## Run tests

```bash
python -m pytest tests/test_budget_api.py -v
```
