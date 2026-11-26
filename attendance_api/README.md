# Attendance API

A tiny REST attendance tracker built with Flask. Students live in an
in-memory store (fresh on every app start — nothing is written to
disk). Mark each student present, absent, or late on a date and the
API computes per-student attendance percentages plus a class summary.

## Why

Attendance is the simplest classroom record there is: who showed up,
when. This app shows the shape of a small stateful API: create, mark,
report, plus derived values (the percentage) computed from the stored
marks. Present and late both count as attended days.

## Usage

```bash
pip install -r requirements.txt
python -m attendance_api.demo
```

The demo creates three students, marks a few days each, and prints
each student's percentage plus the class summary.

To run the app for real:

```bash
flask --app "attendance_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/students` | Create a student: `{"name": str}` |
| GET | `/students` | List all students with their marks |
| GET | `/students/<id>` | Fetch one student |
| POST | `/students/<id>/marks` | Mark a day: `{"date": "YYYY-MM-DD", "status": "present" \| "absent" \| "late"}` |
| GET | `/students/<id>/attendance` | Per-student report with percentage |
| GET | `/summary` | Class-wide summary with overall percentage |

A student looks like
`{"id": 1, "name": "Ayesha", "marks": {"2026-11-23": "present"},
"total_marks": 1, "attendance_percentage": 100.0}`.

Validation errors return `400` with `{"error": "..."}`; unknown
students and routes return `404`. Marking the same date twice returns
`409`; an invalid status or date returns `400`.

## Demo output

```
Created student: Ayesha (id 1)
Created student: Bilal (id 2)
Created student: Chen (id 3)
Ayesha: 2 present, 0 late, 1 absent over 3 days -> 66.67% attendance
Bilal: 2 present, 1 late, 0 absent over 3 days -> 100.0% attendance
Chen: 0 present, 0 late, 2 absent over 2 days -> 0.0% attendance
Class summary: 3 students, 8 marks, overall 62.5% attendance
```

## Run tests

```bash
python -m pytest tests/test_attendance_api.py -v
```
