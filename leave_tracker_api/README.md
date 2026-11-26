# Leave Tracker API

A tiny REST leave tracker built with Flask. Employees and their leave
requests live in an in-memory store (fresh on every app start —
nothing is written to disk). Employees get a balance per leave type
(annual, sick, casual), request time off, and a manager approves or
rejects each request; approval deducts the days from the balance.

## Why

Leave is a small approval workflow: request, check the balance, check
for overlaps, decide. This app shows that shape — a pending request
that only becomes real (deducted) once approved, with the two classic
rejections: not enough balance left, and dates that clash with an
existing request.

## Usage

```bash
pip install -r requirements.txt
python -m leave_tracker_api.demo
```

The demo creates an employee with 20 annual days, requests 5 days,
approves them, and shows 15 days remaining.

To run the app for real:

```bash
flask --app "leave_tracker_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/employees` | Create an employee: `{"name": str, "balances": {"annual": int, "sick": int, "casual": int}?}` (defaults 20/10/7) |
| GET | `/employees` | List all employees with balances |
| GET | `/employees/<id>` | Fetch one employee |
| GET | `/employees/<id>/balance` | Remaining balance per leave type |
| POST | `/employees/<id>/leave` | Request leave: `{"type": "annual" \| "sick" \| "casual", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "reason": str?}` |
| GET | `/employees/<id>/leave` | List an employee's requests |
| GET | `/leave/<id>` | Fetch one request |
| POST | `/leave/<id>/approve` | Approve a pending request (deducts days) |
| POST | `/leave/<id>/reject` | Reject a pending request |

Leave days are counted inclusively (2026-12-07 to 2026-12-11 is 5
days). A request looks like
`{"id": 1, "employee_id": 1, "type": "annual", "start_date":
"2026-12-07", "end_date": "2026-12-11", "days": 5, "reason": "Winter
break", "status": "pending"}`.

Validation errors return `400` with `{"error": "..."}`; unknown
employees, requests, and routes return `404`. A request overlapping
a pending or approved request returns `409`; a request for more days
than the remaining balance returns `400`; deciding an already-decided
request returns `409`.

## Demo output

```
Created employee: Sara (id 1), balances {'annual': 20, 'casual': 7, 'sick': 10}
Requested 5 annual days (2026-12-07 to 2026-12-11): pending
Request 1 approved
Remaining balances: {'annual': 15, 'casual': 7, 'sick': 10} (annual 15 of 20 left)
```

## Run tests

```bash
python -m pytest tests/test_leave_tracker_api.py -v
```
