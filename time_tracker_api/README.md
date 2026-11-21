# time_tracker_api

A small Flask time tracker: projects with an hourly rate, and time
entries you start/stop like a stopwatch or add by hand, kept in
memory (a fresh store per `create_app()` call, nothing written to
disk). One timer can run per project at a time; completed time is
turned into earnings at the project's hourly rate.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/projects` | Create a project: `{"name", "hourly_rate"?, "description"?}` |
| GET | `/projects` | List projects |
| GET | `/projects/<id>` | Fetch one project |
| PUT | `/projects/<id>` | Update name, hourly_rate and/or description |
| DELETE | `/projects/<id>` | Delete a project and its entries |
| POST | `/projects/<id>/entries/start` | Start a timer: `{"description"?, "start_time"?}` (defaults to now) |
| POST | `/entries/<id>/stop` | Stop a running timer: `{"end_time"?}` (defaults to now) |
| POST | `/projects/<id>/entries` | Add a finished entry: `{"start_time", "end_time", "description"?}` |
| GET | `/projects/<id>/entries` | List a project's entries |
| GET | `/entries/<id>` | Fetch one entry |
| DELETE | `/entries/<id>` | Delete an entry |
| GET | `/projects/<id>/summary` | Total/completed time, entry counts, running flag, earnings |
| GET | `/summary` | Totals across all projects |

A project is `{"id", "name", "hourly_rate", "description"}`. An entry
is `{"id", "project_id", "description", "start_time", "end_time",
"duration_minutes", "running"}`. Times are ISO datetimes like
`2026-11-21T09:00`. One project per name (case-insensitive).

Validation: a missing/empty `name`, a negative `hourly_rate`, a time
that is not an ISO datetime, or an `end_time` that is not after
`start_time` returns JSON `400`; an unknown project or entry returns
JSON `404`; a duplicate project name, or starting a second timer on
a project that already has one running, returns JSON `409`.

## Run

```bash
python time_tracker_api/demo.py
python -m pytest tests/test_time_tracker_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/projects -H 'Content-Type: application/json' \
  -d '{"name": "NovaStack", "hourly_rate": 15}'

curl -X POST localhost:5000/projects/1/entries/start -H 'Content-Type: application/json' \
  -d '{"description": "Landing page"}'

curl -X POST localhost:5000/entries/1/stop -H 'Content-Type: application/json' -d '{}'
curl localhost:5000/projects/1/summary
curl localhost:5000/summary
```
