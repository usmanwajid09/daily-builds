# Chore Tracker API

A tiny REST API for household chores built with Flask. Chores live in
an in-memory store (fresh on every app start — nothing is written to
disk).

## Why

Shared chores cause arguments when nobody remembers whose turn it is.
This app rotates the turn automatically every time a chore is marked
done, and flags chores whose due date has passed.

## Usage

```bash
pip install -r requirements.txt
python chore_tracker_api/demo.py
```

## Endpoints

- `GET /chores` — list all chores with current assignee + overdue flag
- `POST /chores` — create a chore (`{"name", "assignees", "frequency_days"?}`)
- `GET /chores/<id>` — fetch one chore
- `POST /chores/<id>/done` — mark done; rotates to the next assignee
- `PUT /chores/<id>` — update name / assignees / frequency
- `DELETE /chores/<id>` — delete a chore
```

## Tests

```bash
python -m pytest tests/test_chore_tracker_api.py -q
```
