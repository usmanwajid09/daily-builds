# Gift Ideas API

A tiny REST API for gift ideas built with Flask. People and gift ideas
live in an in-memory store (fresh on every app start — nothing is
written to disk).

## Why

Birthdays sneak up and gift budgets get blown. This app keeps gift
ideas per person with a price and a status (`idea` / `bought`), so you
can see at a glance what is left of each person's budget.

## Usage

```bash
pip install -r requirements.txt
python gift_ideas_api/demo.py
```

## Endpoints

- `GET /people` / `POST /people` — list / add a person
- `GET /people/<id>` — one person, with budget summary
- `GET /people/<id>/budget` — budget, spent, remaining
- `GET /people/<id>/ideas` / `POST /people/<id>/ideas` — list / add a gift idea
- `PUT /ideas/<id>` — update title / price / occasion / status
- `DELETE /ideas/<id>` — delete a gift idea
```

## Tests

```bash
python -m pytest tests/test_gift_ideas_api.py -q
```
