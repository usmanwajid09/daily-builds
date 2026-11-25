# Study Planner API

A tiny REST study planner API built with Flask. Subjects and study
sessions live in an in-memory store (fresh on every app start — nothing
is written to disk), so it is ideal for learning REST patterns and for
testing with Flask's test client.

## Why

A study planner is a small CRUD app with two aggregate views: totals per
subject (total minutes, session count, days studied, average per day)
and a daily goal check that compares each subject's studied minutes
against its daily goal.

## Usage

```bash
pip install -r requirements.txt
python study_planner_api/demo.py
```

The demo creates two subjects, logs four study sessions, prints the
Maths totals and the daily goal check for 2026-11-25, and shows a
negative session length being rejected.

To run the app for real:

```bash
flask --app "study_planner_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/subjects` | Create a subject: `{"name", "daily_goal_minutes"}` |
| GET | `/subjects` | List all subjects |
| GET | `/subjects/<id>` | Fetch one subject |
| DELETE | `/subjects/<id>` | Delete a subject (and its sessions) |
| POST | `/sessions` | Log a session: `{"subject_id", "minutes", "date"? (YYYY-MM-DD), "topic"?}` |
| GET | `/sessions` | List sessions; `?subject_id=` and `?date=` filters |
| DELETE | `/sessions/<id>` | Delete a session |
| GET | `/subjects/<id>/totals` | Per-subject totals: total minutes, session count, days studied |
| GET | `/goals/daily` | Daily goal check; `?date=YYYY-MM-DD` (defaults to today) |

A session looks like
`{"id": 1, "subject_id": 1, "date": "2026-11-25", "minutes": 40, "topic": "Algebra"}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
return `404`; the goal check reports `met` per subject and `all_met`
overall.

## Run tests

```bash
python -m pytest tests/test_study_planner_api.py -v
```
