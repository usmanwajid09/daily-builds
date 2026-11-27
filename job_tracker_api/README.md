# Job Application Tracker API

A small Flask API for tracking job applications through the hiring
pipeline (`applied` → `screening` → `interview` → `offer` /
`rejected`), with per-status stats, offer/interview rates, and a list
of upcoming interviews. In-memory store — nothing is written to disk,
and each `create_app()` call starts empty.

## Endpoints

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/applications` | Create an application (`company`, `role`, `status?`, `applied_date?`, `interview_date?`, `salary?`, `notes?`) |
| GET | `/applications` | List applications; `?status=` and `?company=` filters |
| GET | `/applications/<id>` | Fetch one application |
| PUT | `/applications/<id>` | Update company / role / dates / salary / notes (status changes go through the status endpoint) |
| DELETE | `/applications/<id>` | Delete an application |
| POST | `/applications/<id>/status` | Move an application through the pipeline; backwards moves and moves out of `rejected` are 409 |
| GET | `/stats` | Total, counts per status, offer rate and interview rate |
| GET | `/interviews/upcoming` | Applications in `interview` status with an interview date today or later, sorted by date |

Pipeline moves: `applied` → `screening` / `interview` / `offer` /
`rejected`, `screening` → `interview` / `offer` / `rejected`,
`interview` → `offer` / `rejected`, `offer` → `rejected`
(declined offer). `rejected` is terminal.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python job_tracker_api/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_job_tracker_api.py -q
```

## Example output

```
Applied #1: Associate Software Engineer at Acme Corp (PKR 90,000/month)
Applied #2: AI/ML Engineer at Globex (PKR 120,000/month)
Applied #3: Web Developer at Initech (PKR 80,000/month)
Applied #4: Backend Developer at Umbrella (PKR 100,000/month)
Moved #1 to screening
Moved #2 to interview, interview 2026-12-05
Moved #3 to offer
Moved #4 to rejected
Moving a rejected application rejected: 409 {'error': 'cannot move an application from rejected to interview'}
Stats: 4 applications, by status {'applied': 0, 'interview': 1, 'offer': 1, 'rejected': 1, 'screening': 1}, offer rate 25.0%, interview rate 25.0%
Upcoming interviews: 1
  2026-12-05: AI/ML Engineer at Globex
```
