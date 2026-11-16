# Workout Tracker API

A tiny REST workout tracker API built with Flask. Workouts live in an
in-memory store (fresh on every app start — nothing is written to
disk), so it is ideal for learning REST patterns and for testing with
Flask's test client.

## Why

A workout log is a small CRUD app with two aggregate views: totals for
a calendar week (Monday to Sunday) and totals grouped by workout type.
This app shows how to work with dates, group records, and compute
totals without a database.

## Usage

```bash
pip install -r requirements.txt
python workout_api/demo.py
```

The demo logs five workouts across two weeks, prints the weekly totals
for the week of 2026-11-16, prints the per-type summary, and shows a
negative duration being rejected.

To run the app for real:

```bash
flask --app "workout_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/workouts` | List all workouts; `?type=run` filters by type |
| POST | `/workouts` | Log a workout: `{"type", "duration_minutes", "calories", "date"?}` |
| GET | `/workouts/<id>` | Fetch one workout |
| DELETE | `/workouts/<id>` | Delete a workout |
| GET | `/workouts/weekly` | Totals for a week; `?week=YYYY-MM-DD` picks any day in the week (defaults to the current week) |
| GET | `/workouts/summary` | Totals grouped by workout type |

A workout looks like
`{"id": 1, "type": "run", "duration_minutes": 30, "calories": 320, "date": "2026-11-16"}`.
`date` defaults to today when omitted. Weekly totals look like
`{"week_start": "2026-11-16", "week_end": "2026-11-22", "count": 4, "total_duration_minutes": 160, "total_calories": 1330}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
and routes return `404` with `{"error": "..."}`.

## Run tests

```bash
python -m pytest tests/test_workout_api.py -v
```
