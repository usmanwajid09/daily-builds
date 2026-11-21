# pomodoro_api

Flask pomodoro timer API: create focus / break sessions, start and
complete them, cancel them, and read overall and per-day stats.
In-memory store, fresh for each app instance.

## Endpoints

- `POST /sessions` — create a session: `{"task": str, "type":
  "focus" | "short_break" | "long_break"?, "duration_minutes": int?}`.
  Defaults: focus, 25 min for focus, 5 for short break, 15 for long
  break. Duration must be 1–240 minutes.
- `GET /sessions` — list sessions; `?status=`, `?type=` and
  `?date=YYYY-MM-DD` filters.
- `GET /sessions/<id>` — fetch one session.
- `POST /sessions/<id>/start` — start a pending session (409 unless
  pending).
- `POST /sessions/<id>/complete` — complete a running session
  (409 unless running).
- `POST /sessions/<id>/cancel` — cancel a pending or running session
  (409 if already completed / cancelled).
- `DELETE /sessions/<id>` — delete a session.
- `GET /stats` — totals, counts by type and status, completed focus
  minutes, today's counts.
- `GET /stats/daily?date=YYYY-MM-DD` — stats for one day (today if
  no date given).

A session is `{"id", "task", "type", "duration_minutes", "status",
"started_at", "completed_at", "created_at"}`; timestamps are UTC ISO
8601 strings. Bad input returns 400, missing sessions 404, invalid
state transitions 409.

## Run

```bash
python -m pytest tests/test_pomodoro_api.py -q
python pomodoro_api/demo.py
```

## Example output

```
Created focus session #1: 'Write report' (25 min, status pending)
Created break session #2: 'Stretch' (5 min)
Created custom session #3: 50 min focus
Started session #1: status running, started_at 2026-11-21T18:00:00.123456Z
Completed session #1: status completed, completed_at 2026-11-21T18:00:00.234567Z
Restarting a completed session rejected: 409 {'error': 'only a pending session can be started (current status: completed)'}
Cancelled session #2: status cancelled
Stats: 3 sessions, 1 completed, 25 focus minutes completed, by status {'cancelled': 1, 'completed': 1, 'pending': 1, 'running': 0}
Today (2026-11-21): 1 completed, 25 focus minutes
```
