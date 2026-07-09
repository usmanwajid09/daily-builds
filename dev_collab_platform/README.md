# dev-collab-platform

A real-time developer collaboration platform: multi-tenant auth, a
workspace/project data model, and a live task board pushed to every
connected client over a **hand-rolled WebSocket server** (no
`websockets` / `flask-socketio` / any WS library -- the RFC 6455
handshake and frame protocol are implemented from raw TCP sockets in
`app/ws/protocol.py`, in the same spirit as this repo's other
from-scratch protocol/algorithm implementations).

This is Milestone 1 of the 2-milestone `dev-collab-platform` arc.
Milestone 2 will add comments/mentions, notifications, and richer
roles/permissions on top of what's built here.

## What's here

- **Auth**: signup/login, bcrypt password hashing, HS256 JWTs
  (`app/auth.py`).
- **Multi-tenant data model** (`app/db.py`): `workspaces` + a global
  `users` table joined through `memberships` (so one person can belong
  to more than one workspace, with a different role in each) +
  `projects` + `tasks`. Every workspace-scoped query filters on
  `workspace_id`, and that id is always the one embedded in the
  verified JWT -- never a client-supplied field -- so there's no
  request parameter that lets one workspace read or write another's
  data. The same rule is enforced on the WebSocket side (see below).
- **REST API** (`app/routes/`): workspace signup/login, workspace
  info + invites, project CRUD, and a task board (`/api/projects/<id>/tasks`
  grouped by column, plus create/patch/delete on individual tasks).
- **Real-time task board** (`app/ws/`): a threaded WebSocket server
  (`server.py`) built on the hand-written protocol implementation
  (`protocol.py`). Clients authenticate with the same JWT the REST API
  issues, subscribe to a project's channel, and receive
  `task_created` / `task_updated` / `task_deleted` events the instant
  the REST API commits that change -- both sides share one
  process-wide `Broadcaster` and one SQLite database.
- **Browser demo** (`static_demo.html`): a single dependency-free HTML
  file using the browser's native `WebSocket` API (no client library
  needed since the wire protocol is standard RFC 6455) that signs
  up/logs in, creates a project, and shows the board updating live.
  Open it in two tabs to see one tab's change show up in the other.

## Why hand-roll the WebSocket layer

The arc description calls for "real-time task board or doc
(websockets)" specifically. `websockets` and `flask-socketio` install
fine in this sandbox, but the more interesting and durable artifact is
understanding *why* a WebSocket connection works: the
Sec-WebSocket-Accept handshake derivation (SHA-1 of the client key +
the RFC's magic GUID, base64-encoded), and the binary frame format
(FIN/opcode byte, 7/16/64-bit length encoding, client-to-server
masking with a 4-byte XOR key). Both are implemented and unit-tested
against the worked example straight from RFC 6455 section 1.3.

## Data model

```
workspaces          users               memberships          projects            tasks
--------             --------            ------------------   --------            --------
id                   id                  id                    id                  id
name                 email (unique)      user_id -> users.id   workspace_id        project_id
slug (unique)        password_hash       workspace_id          name                title
created_at           created_at          role (owner|admin|    created_at          description
                                           member)                                  status (todo|
                                          UNIQUE(user_id,                            in_progress|done)
                                           workspace_id)                            position
                                                                                     created_by
                                                                                     created_at / updated_at
```

## WebSocket protocol (JSON text frames)

```
client -> server
  {"type": "auth", "token": "<jwt>"}                 -- must be sent first
  {"type": "subscribe", "project_id": <int>}          -- must belong to caller's workspace
  {"type": "unsubscribe", "project_id": <int>}

server -> client
  {"type": "auth_ok", "workspace_id": <int>, "role": "<role>"}
  {"type": "subscribed", "project_id": <int>}
  {"type": "unsubscribed", "project_id": <int>}
  {"type": "task_created", "project_id": <int>, "task": {...}}
  {"type": "task_updated", "project_id": <int>, "task": {...}}
  {"type": "task_deleted", "project_id": <int>, "task_id": <int>}
  {"type": "error", "message": "<str>"}
```

## Usage

```bash
pip install -r requirements.txt
python run.py
# REST API on http://127.0.0.1:5000, WebSocket server on ws://127.0.0.1:8765
# then open static_demo.html directly in a browser (two tabs to see live sync)
```

Environment variables (all optional): `DB_PATH`, `JWT_SECRET`,
`HTTP_HOST`, `HTTP_PORT`, `WS_HOST`, `WS_PORT`.

### Example REST flow

```bash
curl -s -X POST http://127.0.0.1:5000/api/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"a@example.com","password":"password123","workspace_name":"Acme"}'
# -> {"token": "...", "workspace_id": 1, "workspace_slug": "acme", "role": "owner"}

curl -s -X POST http://127.0.0.1:5000/api/projects \
  -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \
  -d '{"name": "Launch Plan"}'
```

## Run tests

```bash
python -m pytest tests/
```

79 tests: SQLite data-model/constraint tests, auth unit tests (hashing,
JWT issue/verify/expiry), WebSocket protocol unit tests (handshake
against the RFC's own worked example, frame encode/decode at boundary
lengths including 16-bit and 64-bit length prefixes, masked vs.
unmasked frames, fragmented-frame rejection, truncated-connection
handling), Flask REST route tests (including cross-workspace access
denial), and full end-to-end integration tests that spin up a *real*
`WebSocketServer` on a real socket and a hand-rolled test client
(`tests/ws_test_client.py`, itself dependency-free) to prove a REST
write really pushes a live event to a subscribed socket -- including
multiple simultaneous subscribers and a dead-socket cleanup case.

## Known limitations (honest, not swept under the rug)

- **One thread per WebSocket connection.** Fine for a demo/dev-scale
  board; a production version would move to an event loop
  (asyncio/selectors) instead of `threading.Thread` per connection.
- **No fragmented-frame / continuation-frame support.** The app only
  ever sends small JSON control messages, so `decode_frame` treats a
  fragmented frame as a protocol error rather than reassembling it.
  A general-purpose WebSocket client library would need that.
- **No permessage-deflate / extension negotiation.** Not needed at
  this message size, and it would meaningfully complicate the framing
  code for no real benefit here.
- **Best-effort broadcast.** If a socket write fails mid-broadcast
  (e.g. the peer already disconnected), that socket is dropped from
  the channel silently rather than surfacing an error to the writer
  that triggered the broadcast -- so a REST client that creates a task
  doesn't get blamed for another client's dead connection state
  (`test_disconnect_cleans_up_subscription` in
  `tests/test_ws_integration.py`).
- **Flask dev server, not production WSGI.** `run.py` uses
  `flask_app.run()` for a self-contained demo; a real deployment would
  put this behind gunicorn/uwsgi the way `saas_starter`'s Dockerfile
  does.
