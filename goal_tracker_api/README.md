# Goal Tracker API

A tiny REST API for personal goals built with Flask. Goals live in an
in-memory store (fresh on every app start — nothing is written to disk).

## Why

Goals only work when you can see progress. This app tracks a goal's
target and deadline, breaks it into milestones, and computes overall
progress from the milestones — or from a manual percentage when there
are none yet.

## Usage

```bash
pip install -r requirements.txt
python goal_tracker_api/demo.py
```

## Endpoints

- `GET /goals` — list all goals with computed progress
- `POST /goals` — create a goal (`{"title", "target", "deadline"}`)
- `GET /goals/<id>` — fetch one goal
- `PUT /goals/<id>/progress` — set progress 0–100
- `POST /goals/<id>/milestones` — add a milestone
- `PUT /goals/<id>/milestones/<mid>` — mark a milestone done
- `DELETE /goals/<id>` — delete a goal
```

## Tests

```bash
python -m pytest tests/test_goal_tracker_api.py -q
```
