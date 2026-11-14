# Notes API

A tiny REST notes API built with Flask. Notes live in an in-memory
store (fresh on every app start — nothing is written to disk), with
full-text search and tag filtering on the list endpoint.

## Why

Notes are the classic CRUD example, and search plus tags turn a plain
list into something actually useful. This app shows query-parameter
filtering and partial updates alongside the usual create/read/update/
delete cycle.

## Usage

```bash
pip install -r requirements.txt
python -m notes_api.demo
```

The demo creates two notes, searches by text, filters by tag, updates
one, and deletes it.

To run the app for real:

```bash
flask --app "notes_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/notes` | Create a note: `{"title": str, "body": str?, "tags": [str]?}` |
| GET | `/notes` | List notes; `?search=<text>` matches title/body, `?tag=<tag>` filters by tag |
| GET | `/notes/<id>` | Fetch one note |
| PUT | `/notes/<id>` | Update `title`, `body` and/or `tags` |
| DELETE | `/notes/<id>` | Delete a note |

A note looks like
`{"id": 1, "title": "Grocery list", "body": "Milk, eggs, bread",
"tags": ["home"]}`. Tags are de-duplicated and trimmed on write.

Validation errors return `400` with `{"error": "..."}`; unknown ids and
routes return `404` with `{"error": "..."}`.

## Run tests

```bash
python -m pytest tests/test_notes_api.py -v
```
