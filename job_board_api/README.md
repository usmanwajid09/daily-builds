# job_board_api

A Flask API for a job board: post jobs, apply, and move applications through
a review workflow.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/jobs` | Create a job (`title`, `company`, `location`, `job_type`, `salary_min?`, `salary_max?`, `description?`) |
| GET | `/jobs` | List jobs; filter by `?location=`, `?type=`, `?status=`, `?company=` |
| GET | `/jobs/<id>` | Fetch one job |
| PUT | `/jobs/<id>` | Update a job |
| DELETE | `/jobs/<id>` | Delete a job (and its applications) |
| POST | `/jobs/<id>/close` | Close a posting (blocks new applications) |
| POST | `/jobs/<id>/reopen` | Reopen a posting |
| POST | `/jobs/<id>/applications` | Apply for a job |
| GET | `/jobs/<id>/applications` | Applications for one job |
| GET | `/applications/<id>` | Fetch one application |
| POST | `/applications/<id>/status` | Move an application (`submitted -> reviewed -> shortlisted -> hired`, or `rejected`) |

Job types: `full-time`, `part-time`, `contract`, `internship`, `remote`.
`salary_max` must not be below `salary_min`. One application per email per
job (409 on a duplicate). Validation errors return 400, missing jobs or
applications return 404, and illegal state changes return 409, all as
`{"error": "..."}`.

## How to run

```bash
python job_board_api/demo.py
```

Tests:

```bash
python -m pytest job_board_api/tests/test_job_board_api.py -v
```

## Demo output

```
Posted job #1: Associate Software Engineer at NovaStack (Lahore, full-time) — open
Posted job #2: AI/ML Engineer at TechCorp
Jobs in Lahore: ['Associate Software Engineer']
Application #1 for job #1 — status submitted
Application #1 -> reviewed
Application #1 -> shortlisted
Application #1 -> hired
Job #1 status: closed
Applying to a closed job: 409 (job is closed and no longer accepts applications)
Job #1 status after reopen: open
```
