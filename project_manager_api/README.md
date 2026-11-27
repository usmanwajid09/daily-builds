# Project Manager API

A small Flask API for managing projects and their tasks, with a task
status workflow (`todo` → `in_progress` → `done`), per-project
progress percentages, and overdue flags. In-memory store — nothing is
written to disk, and each `create_app()` call starts empty.

## Endpoints

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/projects` | Create a project (`name`, `description?`); 409 on duplicate name |
| GET | `/projects` | List projects with task counts and progress % |
| GET | `/projects/<id>` | Fetch one project with its tasks |
| PUT | `/projects/<id>` | Rename / change description; 409 on duplicate name |
| DELETE | `/projects/<id>` | Delete a project and all its tasks |
| POST | `/projects/<id>/tasks` | Add a task (`title`, `description?`, `assignee?`, `due_date?` `YYYY-MM-DD`, `status?`) |
| GET | `/projects/<id>/tasks` | List tasks; `?status=` filters; each task has an `overdue` flag |
| GET | `/projects/<id>/tasks/<task_id>` | Fetch one task |
| PUT | `/projects/<id>/tasks/<task_id>` | Update title / description / assignee / due date |
| DELETE | `/projects/<id>/tasks/<task_id>` | Delete a task |
| POST | `/projects/<id>/tasks/<task_id>/status` | Move a task through the workflow; invalid moves are 409 |
| GET | `/projects/<id>/progress` | Total/done counts, progress %, overdue count and ids, counts by status |

Status workflow: `todo` → `in_progress`, `todo` → `done`,
`in_progress` → `done`, `in_progress` → `todo` (reopen),
`done` → `todo` (reopen). Any other move (e.g. `done` →
`in_progress`) is rejected with 409. A task is overdue when its due
date is in the past and it is not done.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python project_manager_api/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_project_manager_api.py -q
```

## Example output

```
Created project #1: 'Website Redesign'
Added task #1: 'Design homepage' (todo, due 2026-12-15, overdue=False)
Added task #2: 'Write copy' (todo, due 2026-12-20, overdue=False)
Added task #3: 'Deploy to staging' (todo, due 2020-01-01, overdue=True)
Moved 'Design homepage' to in_progress
Moved 'Design homepage' to done
done -> in_progress rejected (workflow): 409 {'error': 'cannot move a task from done to in_progress'}
Progress: 1/3 done (33.3%), 1 overdue (ids [3]), by status {'done': 1, 'in_progress': 0, 'todo': 2}
```
