# Gym Membership API

A tiny REST gym membership API built with Flask. Plans, members and
check-ins live in an in-memory store (fresh on every app start —
nothing is written to disk), so it is ideal for learning REST patterns
and for testing with Flask's test client.

## Why

A gym membership is a small CRUD app with a date rule: a member's
expiry date is the start date plus the plan's duration, and an expired
member cannot check in. Each member view carries an `expired` flag and
a `days_remaining` count.

## Usage

```bash
pip install -r requirements.txt
python gym_membership_api/demo.py
```

The demo creates a 30-day plan, adds a current member and one whose
membership started 2026-09-01, checks the current member in, shows a
same-day repeat and an expired check-in being rejected, and lists the
expired members.

To run the app for real:

```bash
flask --app "gym_membership_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/plans` | Create a plan: `{"name", "duration_days", "price"}` |
| GET | `/plans` | List all plans |
| GET | `/plans/<id>` | Fetch one plan |
| POST | `/members` | Add a member: `{"name", "plan_id", "start_date"? (YYYY-MM-DD, defaults to today)}` |
| GET | `/members` | List all members (with `expired` and `days_remaining`) |
| GET | `/members/expired` | List only expired members |
| GET | `/members/<id>` | Fetch one member |
| DELETE | `/members/<id>` | Delete a member (and their check-ins) |
| POST | `/members/<id>/check-in` | Check in: `{"date"?}` — 409 if expired or already checked in that day |
| GET | `/members/<id>/check-ins` | List a member's check-ins with a count |

A member looks like
`{"id": 1, "name": "Usman", "plan_id": 1, "start_date": "2026-11-25", "expiry_date": "2026-12-25", "expired": false, "days_remaining": 30, "checkin_count": 0}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
return `404`; an expired or duplicate check-in returns `409`.

## Run tests

```bash
python -m pytest tests/test_gym_membership_api.py -v
```
