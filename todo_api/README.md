# Todo API

A tiny REST todo-list API built with Flask. Todos live in an in-memory
store (fresh on every app start — nothing is written to disk), so it is
ideal for learning REST patterns and for testing with Flask's test
client.

## Why

CRUD over HTTP is the backbone of most web backends. This app shows the
shape of it — JSON in, JSON out, proper status codes — without a
database getting in the way.

## Usage

```bash
pip install -r requirements.txt
python -m todo_api.demo
```

The demo creates two todos, lists them, marks one complete, and prints
every JSON response along the way.

To run the app for real:

```bash
flask --app "todo_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/todos` | List all todos |
| POST | `/todos` | Create a todo: `{"title": str, "done": bool?}` |
| GET | `/todos/<id>` | Fetch one todo |
| PUT | `/todos/<id>` | Update `title` and/or `done` |
| DELETE | `/todos/<id>` | Delete a todo |

Validation errors return `400` with `{"error": "..."}`; unknown ids and
routes return `404` with `{"error": "..."}`. A todo looks like
`{"id": 1, "title": "Buy milk", "done": false}`.

## Run tests

```bash
python -m pytest tests/test_todo_api.py -v
```
