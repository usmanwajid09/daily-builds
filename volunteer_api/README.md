# Volunteer Events API

A tiny REST volunteer events API built with Flask. Events and signups
live in an in-memory store (fresh on every app start — nothing is
written to disk), so it is ideal for learning REST patterns and for
testing with Flask's test client.

## Why

A volunteer event is a small CRUD app with a capacity rule: signups are
rejected once the event is full, and the same volunteer cannot sign up
twice. Volunteers log the hours they worked, and the per-volunteer
endpoint totals hours across every event they joined.

## Usage

```bash
pip install -r requirements.txt
python volunteer_api/demo.py
```

The demo creates a food drive with capacity 2, signs up two
volunteers, shows a third signup being rejected with 409, logs 4 hours
for the first volunteer, and prints that volunteer's hour total.

To run the app for real:

```bash
flask --app "volunteer_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/events` | Create an event: `{"name", "date", "capacity", "location"?}` |
| GET | `/events` | List all events (with signup count, spots remaining, total hours) |
| GET | `/events/<id>` | Fetch one event |
| DELETE | `/events/<id>` | Delete an event (and its signups) |
| POST | `/events/<id>/signups` | Sign up: `{"volunteer_name", "email"?}` — 409 if full or already signed up |
| GET | `/events/<id>/signups` | List an event's signups |
| POST | `/events/<id>/signups/<signup_id>/hours` | Log hours: `{"hours"}` (adds to the running total) |
| GET | `/volunteers` | Per-volunteer totals: `?name=` — total hours, signup count, events |

An event looks like
`{"id": 1, "name": "Food Drive", "date": "2026-11-29", "capacity": 2, "location": "Lahore", "signup_count": 2, "spots_remaining": 0, "total_hours": 4}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
return `404`; a full event or a duplicate signup returns `409`.

## Run tests

```bash
python -m pytest tests/test_volunteer_api.py -v
```
