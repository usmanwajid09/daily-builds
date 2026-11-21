# sleep_tracker_api

A small Flask sleep log API: log nights with bedtime, wake time and a
1–5 quality rating, kept in memory (a fresh store per `create_app()`
call, nothing written to disk). Sleep duration is computed from the
times and may cross midnight (e.g. 23:30 → 07:15 is 465 minutes).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/nights` | Log a night: `{"date": "YYYY-MM-DD", "bedtime": "HH:MM", "wake_time": "HH:MM", "quality": 1-5, "notes"?}` |
| GET | `/nights` | List nights, newest first; `?date=` and `?min_quality=` filters |
| GET | `/nights/<id>` | Fetch one night |
| PUT | `/nights/<id>` | Update date, bedtime, wake_time, quality and/or notes |
| DELETE | `/nights/<id>` | Delete a night |
| GET | `/stats` | Totals, averages, longest/shortest night, consecutive-day streak |

A night is `{"id", "date", "bedtime", "wake_time",
"duration_minutes", "duration_hours", "quality", "notes"}`. One entry
per date: logging a second night for the same date is refused. The
streak counts consecutive logged days ending at the most recent date.

Validation: a date that is not `YYYY-MM-DD`, a time that is not
`HH:MM`, equal bedtime and wake time (0 minutes), or a quality outside
1–5 returns JSON `400`; an unknown night returns JSON `404`; a
duplicate date returns JSON `409`.

## Run

```bash
python sleep_tracker_api/demo.py
python -m pytest tests/test_sleep_tracker_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/nights -H 'Content-Type: application/json' \
  -d '{"date": "2026-11-21", "bedtime": "23:30", "wake_time": "07:15", "quality": 4}'

curl 'localhost:5000/nights?min_quality=4'
curl localhost:5000/stats
```
