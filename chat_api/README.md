# chat_api

A small Flask chat API with rooms and per-room message history, kept in
memory (a fresh store per `create_app()` call, nothing written to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/rooms` | Create a room: `{"name": str}` |
| GET | `/rooms` | List rooms (each with `message_count`) |
| GET | `/rooms/<id>` | Fetch one room |
| DELETE | `/rooms/<id>` | Delete a room and all its messages |
| POST | `/rooms/<id>/messages` | Post a message: `{"sender": str, "text": str}` |
| GET | `/rooms/<id>/messages` | History, oldest first; `?sender=` filters, `?limit=` caps |
| DELETE | `/rooms/<id>/messages/<message_id>` | Delete one message |

Validation: missing/empty `name`, `sender` or `text` returns JSON
`400`; unknown room or message returns JSON `404`.

## Run

```bash
python chat_api/demo.py
python -m pytest tests/test_chat_api.py -v
```
