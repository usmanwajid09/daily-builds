# ticket_api

A small Flask support-ticket API with a guarded status workflow,
priority and assignee, kept in memory (a fresh store per `create_app()`
call, nothing written to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/tickets` | Create a ticket: `{"title", "description"?, "priority"?, "assignee"?}` |
| GET | `/tickets` | List tickets; `?status=`, `?priority=`, `?assignee=` filters |
| GET | `/tickets/<id>` | Fetch one ticket |
| PUT | `/tickets/<id>` | Update title, description, priority and/or assignee |
| POST | `/tickets/<id>/status` | Move a ticket: `{"status": str}` |
| DELETE | `/tickets/<id>` | Delete a ticket |

New tickets start `open` with default priority `medium`.

## Status workflow

```
open -> in_progress -> resolved -> closed
open -> closed            (direct close allowed)
in_progress -> open       (send back)
resolved -> in_progress   (reopen)
closed -> (terminal)
```

Any other transition returns JSON `400` naming the allowed moves.
`PUT` refuses a `status` field — status changes go through `/status`.

Validation: missing/empty `title`, a bad `priority` or `status`, or a
bad `assignee` returns JSON `400`; an unknown ticket returns JSON `404`.

## Run

```bash
python ticket_api/demo.py
python -m pytest tests/test_ticket_api.py -v
```
