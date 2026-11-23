# Book Club API

A tiny REST API for a book club built with Flask. Books, members,
meetings, votes, and RSVPs live in an in-memory store (fresh on every
app start — nothing is written to disk).

## Why

Picking the next book by group chat is chaos. This app gives every
member one vote (changeable), ranks books by votes, and tracks who is
coming to each meeting.

## Usage

```bash
pip install -r requirements.txt
python book_club_api/demo.py
```

## Endpoints

- `GET /books` / `POST /books` — list / add a book
- `GET /books/<id>` — one book, with its vote count
- `POST /books/<id>/vote` — vote for a book (`{"member_id": n}`)
- `GET /votes` — vote leaderboard, most-voted first
- `GET /members` / `POST /members` — list / add a member
- `GET /meetings` / `POST /meetings` — list / schedule a meeting
- `POST /meetings/<id>/rsvp` — RSVP (`{"member_id": n, "attending": bool}`)
```

## Tests

```bash
python -m pytest tests/test_book_club_api.py -q
```
