# Journal API

A tiny REST personal journal built with Flask. Entries live in an
in-memory store (fresh on every app start — nothing is written to
disk). Each entry has a date, a mood from 1 to 5, some text, and
optional tags. List entries by date range, get the average mood over
a range, and search by tag.

## Why

A journal is a small time-series: dated records with a numeric score
and free-form tags. This app shows that shape — range queries, an
aggregate (average mood) computed over a range, and tag search — with
strict validation on the mood scale and the date format.

## Usage

```bash
pip install -r requirements.txt
python -m journal_api.demo
```

The demo writes four entries over a week, prints the average mood
(4.0 over 4 entries), and searches the "work" tag.

To run the app for real:

```bash
flask --app "journal_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/entries` | Create an entry: `{"date": "YYYY-MM-DD", "mood": 1-5, "text": str, "tags": [str]?}` |
| GET | `/entries` | List entries, optionally filtered by `?from=YYYY-MM-DD&to=YYYY-MM-DD` |
| GET | `/entries/<id>` | Fetch one entry |
| DELETE | `/entries/<id>` | Delete an entry |
| GET | `/entries/search?tag=<tag>` | Entries carrying a tag (case-insensitive) |
| GET | `/stats/mood-average` | Average mood over an optional `?from=...&to=...` range |

An entry looks like
`{"id": 1, "date": "2026-11-20", "mood": 4, "text": "Shipped the
first Docker image.", "tags": ["work", "docker"]}` and the mood
statistic looks like `{"count": 4, "average_mood": 4.0}` (with
`average_mood: null` when there are no entries in range).

Validation errors return `400` with `{"error": "..."}`; unknown
entries and routes return `404`. A mood outside 1–5 or an invalid
date returns `400`.

## Demo output

```
Entry #1 2026-11-20 mood=4 tags=['work', 'docker']
Entry #2 2026-11-22 mood=3 tags=['work']
Entry #3 2026-11-24 mood=5 tags=['health', 'study']
Entry #4 2026-11-26 mood=4 tags=['work', 'study']
Average mood 20–26 Nov: 4.0 over 4 entries
Tag 'work' matched 3 entries: ['2026-11-20', '2026-11-22', '2026-11-26']
Entries 23–26 Nov: 2
```

## Run tests

```bash
python -m pytest tests/test_journal_api.py -v
```
