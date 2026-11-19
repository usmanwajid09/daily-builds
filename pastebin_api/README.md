# Pastebin API

A tiny snippet-sharing API built with Flask. Snippets live in an
in-memory store (fresh on every app start — nothing is written to
disk), so it is ideal for learning REST patterns and for testing with
Flask's test client. Snippets can have an optional expiry in
seconds; expired snippets are gone, not just hidden.

## Usage

```bash
pip install -r requirements.txt
python -m pastebin_api.demo
```

The demo creates a permanent snippet and one that expires after 1
second, lists the live snippets, waits for the expiry, and shows the
expired fetch returning 410 Gone.

To run the app for real:

```bash
flask --app "pastebin_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/snippets` | List live (non-expired) snippets |
| POST | `/snippets` | Create one: `{"content": str, "expires_in": int?}` (seconds) |
| GET | `/snippets/<id>` | Fetch one snippet; expired returns 410 |
| DELETE | `/snippets/<id>` | Delete a snippet |

Validation errors return `400` with `{"error": "..."}`; unknown ids
return `404`; expired snippets return `410` with
`{"error": "snippet N has expired"}`. A snippet looks like
`{"id": 1, "content": "print('hi')", "created_at": 1763530000.0,
"expires_at": null}` (`expires_at` is a Unix timestamp, or `null`
for a permanent snippet).

## Run tests

```bash
python -m pytest tests/test_pastebin_api.py -v
```
