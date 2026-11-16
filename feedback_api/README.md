# Feedback API

A tiny REST feedback API built with Flask. Feedback entries live in an
in-memory store (fresh on every app start — nothing is written to
disk), so it is ideal for learning REST patterns and for testing with
Flask's test client.

## Why

Feedback is a small write-heavy API with one read-heavy twist: an
aggregate summary (average rating and a per-rating distribution) that
the server computes on the fly. This app shows how to validate a
bounded rating, filter a list by query parameter, and build a summary
endpoint from the same store.

## Usage

```bash
pip install -r requirements.txt
python feedback_api/demo.py
```

The demo submits four feedback entries, prints the summary (count,
average rating, distribution), lists the 5-star entries, and shows a
rating of 6 being rejected.

To run the app for real:

```bash
flask --app "feedback_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/feedback` | List all feedback; `?rating=5` filters to one rating |
| POST | `/feedback` | Submit feedback: `{"name"?, "rating": 1-5, "comment"}` |
| GET | `/feedback/summary` | Count, average rating, per-rating distribution |
| GET | `/feedback/<id>` | Fetch one feedback entry |
| DELETE | `/feedback/<id>` | Delete a feedback entry |

A feedback entry looks like
`{"id": 1, "name": "Ayesha", "rating": 5, "comment": "Brilliant service."}`.
The summary looks like
`{"count": 4, "average_rating": 4.0, "distribution": {"1": 0, "2": 1, "3": 0, "4": 1, "5": 2}}`
(`average_rating` is `null` when there is no feedback yet).
Validation errors return `400` with `{"error": "..."}`; unknown ids
and routes return `404` with `{"error": "..."}`.

## Run tests

```bash
python -m pytest tests/test_feedback_api.py -v
```
