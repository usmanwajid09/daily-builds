# Library API

A tiny REST library API built with Flask. Books and loans live in an
in-memory store (fresh on every app start — nothing is written to
disk), so it is ideal for learning REST patterns and for testing with
Flask's test client.

## Why

A library is CRUD plus state: borrowing a copy reduces availability,
returning it restores it, and every loan carries a due date that can go
overdue. This app shows how to model that state with plain dicts and
keep it consistent across endpoints.

## Usage

```bash
pip install -r requirements.txt
python library_api/demo.py
```

The demo adds a book with 2 copies, borrows both, shows the third
borrow being rejected, returns one copy, and lists the loans.

To run the app for real:

```bash
flask --app "library_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/books` | List all books |
| POST | `/books` | Add a book: `{"title", "author", "isbn"?, "total_copies"}` |
| GET | `/books/<id>` | Fetch one book |
| PUT | `/books/<id>` | Update a book |
| DELETE | `/books/<id>` | Delete a book (only when no copies are on loan) |
| POST | `/books/<id>/borrow` | Borrow a copy: `{"borrower", "due_days"?}` (default 14) |
| POST | `/books/<id>/return` | Return a copy: `{"loan_id"}` |
| GET | `/loans` | List all loans; `?overdue=true` shows overdue only |

A book looks like
`{"id": 1, "title": "Dune", "author": "Frank Herbert", "isbn": "9780441172719", "total_copies": 2, "available_copies": 2}`.
A loan looks like
`{"loan_id": 1, "book_id": 1, "borrower": "Ayesha", "borrowed_on": "2026-11-16", "due_on": "2026-11-30", "returned": false, "overdue": false}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
and routes return `404` with `{"error": "..."}`.

## Run tests

```bash
python -m pytest tests/test_library_api.py -v
```
