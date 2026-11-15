# quote_api

A small Flask REST API for storing quotes: add, list, fetch a random
one, and filter by author or tag. All state lives in memory; nothing
is written to disk.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/quotes` | List quotes. Optional `?author=<name>` and `?tag=<tag>`. |
| POST | `/quotes` | Add a quote: `{"text": str, "author": str, "tags": [str]?}` |
| GET | `/quotes/random` | A random quote (404 when no quotes are stored). |
| GET | `/quotes/<id>` | Fetch one quote. |
| DELETE | `/quotes/<id>` | Delete a quote. |

Invalid input returns `400` with a JSON `{"error": ...}` message;
unknown ids return `404`.

## Run

```bash
python quote_api/demo.py
python -m pytest tests/test_quote_api.py -q
```

## Example

```
Added quote: {'id': 1, 'text': 'Simplicity is the soul of efficiency.', 'author': 'Austin Freeman', 'tags': ['simplicity']}
Quotes by Torvalds: [{'id': 2, 'text': 'Talk is cheap. Show me the code.', 'author': 'Linus Torvalds', 'tags': ['code', 'programming']}]
Random quote: {'id': 1, 'text': 'Simplicity is the soul of efficiency.', 'author': 'Austin Freeman', 'tags': ['simplicity']}
```
