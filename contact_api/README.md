# Contact API

A tiny contact-form API built with Flask. Messages live in an
in-memory store (fresh on every app start — nothing is written to
disk), so it is ideal for learning REST patterns and for testing with
Flask's test client.

## Usage

```bash
pip install -r requirements.txt
python -m contact_api.demo
```

The demo submits two valid messages, shows a bad email being
rejected with a 400, lists the inbox, deletes one message and prints
what remains.

To run the app for real:

```bash
flask --app "contact_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/messages` | List all messages |
| POST | `/messages` | Submit one: `{"name": str, "email": str, "message": str}` |
| GET | `/messages/<id>` | Fetch one message |
| DELETE | `/messages/<id>` | Delete a message |

Names must be non-empty, emails must look like `user@host.tld`,
and messages must be non-empty and at most 5000 characters.
Validation errors return `400` with `{"error": "..."}`; unknown ids
return `404`. A message looks like
`{"id": 1, "name": "Ayesha", "email": "ayesha@example.com",
"message": "Love the site!"}`.

## Run tests

```bash
python -m pytest tests/test_contact_api.py -v
```
