# Habit Tracker

A tiny REST habit tracker built with Flask. Habits live in an in-memory
store (fresh on every app start — nothing is written to disk). Mark a
habit done on a day and the API tracks your streak: the number of
consecutive done days ending at the most recent done date.

## Why

Streaks are the simplest motivation loop there is — don't break the
chain. This app shows the shape of a small stateful API: create, list,
mark, unmark, delete, plus a derived value (the streak) computed from
the stored dates.

## Usage

```bash
pip install -r requirements.txt
python -m habit_tracker.demo
```

The demo creates a habit, marks four consecutive days done, and prints
the growing streak.

To run the app for real:

```bash
flask --app "habit_tracker.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/habits` | Create a habit: `{"name": str, "description": str?}` |
| GET | `/habits` | List all habits with streaks |
| GET | `/habits/<id>` | Fetch one habit |
| POST | `/habits/<id>/done` | Mark a day done: `{"date": "YYYY-MM-DD"}` (defaults to today) |
| DELETE | `/habits/<id>/done` | Unmark a day: `{"date": "YYYY-MM-DD"}` |
| DELETE | `/habits/<id>` | Delete a habit |

A habit looks like
`{"id": 1, "name": "Read 20 pages", "description": "Before bed",
"done_dates": ["2026-11-11"], "total": 1, "streak": 1}`.

Validation errors return `400` with `{"error": "..."}`; unknown ids and
routes return `404` with `{"error": "..."}`. Marking the same day twice,
a future date, or an invalid date returns `400`.

## Run tests

```bash
python -m pytest tests/test_habit_tracker.py -v
```
