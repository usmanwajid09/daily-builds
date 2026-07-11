# dev-collab-platform

A real-time developer collaboration platform: multi-tenant auth, a
workspace/project data model, and a live task board pushed to every
connected client over a **hand-rolled WebSocket server** (no
`websockets` / `flask-socketio` / any WS library -- the RFC 6455
handshake and frame protocol are implemented from raw TCP sockets in
`app/ws/protocol.py`, in the same spirit as this repo's other
from-scratch protocol/algorithm implementations).

This is the complete 2-milestone `dev-collab-platform` arc. Milestone 1
built the auth/workspace/project/task-board foundation and the
hand-rolled WebSocket server; Milestone 2 (this update) adds threaded
task comments with @mentions, a notification system (with its own
real-time push channel), and enforced roles/permissions -- role
changes, member removal, and project deletion are all gated by role,
with explicit protection against ever leaving a workspace with zero
owners.

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
  (Not yet updated for Milestone 2's comments/notifications UI --
  those are exercised via the REST/WS APIs directly, see below.)
- **Comments + @mentions** (`app/routes/comment_routes.py`,
  `app/mentions.py`): threaded comments on a task
  (`/api/tasks/<id>/comments`), broadcast to the task's project channel
  like task events. A comment body can @mention a workspace member by
  email (`cc @alice@example.com`, since this app has no separate
  username handle) -- mentioning a real member creates a notification
  for them; mentioning an email that isn't a workspace member, or
  yourself, is silently a no-op rather than an error.
- **Notifications** (`app/routes/notification_routes.py`): a per-user,
  per-workspace inbox (`/api/notifications`, mark-read, mark-all-read)
  fed by mentions and role changes so far. Real-time delivery reuses
  the *same* `Broadcaster` from the task board -- a notification
  channel is just a string key (`f"user:{user_id}"`) instead of an int
  `project_id`; see "WebSocket protocol" below. A user's notification
  channel is scoped to their identity, not to which workspace-scoped
  token authenticated the connection -- if they belong to multiple
  workspaces they get all of their notifications on one connection,
  same as a real notification bell would.
- **Roles/permissions enforcement**: changing a member's role
  (`PATCH /api/workspace/members/<id>`) and removing a member
  (`DELETE /api/workspace/members/<id>`) are now real endpoints, not
  just data-model fields -- only an owner can grant/revoke the owner
  role, an admin can move people between admin/member but never touch
  an owner, anyone can remove themselves (leave), and the last
  remaining owner in a workspace can never be demoted or removed
  (self-removal included) so a workspace can't end up with no one able
  to manage it. Project deletion (`DELETE /api/projects/<id>`) is
  owner/admin only and cascades to the project's tasks, comments, and
  related notifications via `ON DELETE CASCADE` foreign keys.

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
name                 email (unique)      user_id -> users.id   workspace_id        project_id -> projects.id
slug (unique)        password_hash       workspace_id          name                  (ON DELETE CASCADE)
created_at           created_at          role (owner|admin|    created_at          title
                                           member)                                  description
                                          UNIQUE(user_id,                           status (todo|
                                           workspace_id)                             in_progress|done)
                                                                                     position
                                                                                     created_by
                                                                                     created_at / updated_at

comments                                 notifications
--------                                 --------
id                                       id
task_id -> tasks.id (ON DELETE CASCADE)  user_id -> users.id   (recipient)
author_id -> users.id                    workspace_id -> workspaces.id
body                                     type (mention | role_changed)
created_at / updated_at                  message                (denormalized, human-readable)
                                          task_id -> tasks.id    (ON DELETE CASCADE, nullable)
                                          comment_id -> comments.id (ON DELETE CASCADE, nullable)
                                          actor_id -> users.id   (who triggered it, nullable)
                                          read_at                (nullable)
                                          created_at
```

Deleting a project cascades all the way down: project -> tasks ->
comments -> notifications referencing those tasks/comments, all via
`ON DELETE CASCADE`, so `delete_project()` is one `DELETE` statement
with no manual cleanup pass (`test_deleting_project_cascades_to_tasks_and_comments`
in `tests/test_roles_permissions.py`).

## WebSocket protocol (JSON text frames)

```
client -> server
  {"type": "auth", "token": "<jwt>"}                 -- must be sent first
  {"type": "subscribe", "project_id": <int>}          -- must belong to caller's workspace
  {"type": "unsubscribe", "project_id": <int>}
  {"type": "subscribe_notifications"}                 -- joins this user's own notification channel
  {"type": "unsubscribe_notifications"}

server -> client
  {"type": "auth_ok", "workspace_id": <int>, "role": "<role>"}
  {"type": "subscribed", "project_id": <int>}
  {"type": "unsubscribed", "project_id": <int>}
  {"type": "task_created", "project_id": <int>, "task": {...}}
  {"type": "task_updated", "project_id": <int>, "task": {...}}
  {"type": "task_deleted", "project_id": <int>, "task_id": <int>}
  {"type": "comment_created", "project_id": <int>, "task_id": <int>, "comment": {...}}
  {"type": "comment_deleted", "project_id": <int>, "task_id": <int>, "comment_id": <int>}
  {"type": "project_deleted", "project_id": <int>}
  {"type": "subscribed_notifications"}
  {"type": "unsubscribed_notifications"}
  {"type": "notification", "notification": {...}}      -- pushed on a mention or role change
  {"type": "error", "message": "<str>"}
```

`Broadcaster` (in `app/ws/server.py`) is the same class serving both
task-board channels (`int` project_id keys) and notification channels
(`str` `"user:<id>"` keys) -- it's just a dict keyed on whatever
hashable value you subscribe with, so Milestone 2's notifications
needed zero new broadcast infrastructure, just a new channel-key shape.

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

132 tests: SQLite data-model/constraint tests, auth unit tests
(hashing, JWT issue/verify/expiry), WebSocket protocol unit tests
(handshake against the RFC's own worked example, frame encode/decode
at boundary lengths including 16-bit and 64-bit length prefixes,
masked vs. unmasked frames, fragmented-frame rejection,
truncated-connection handling), mention-parsing unit tests (case
folding, dedup, trailing-punctuation handling), Flask REST route tests
(including cross-workspace access denial), comments/mentions tests
(authorship-gated delete, non-member/self-mentions being silent
no-ops, cascade-on-task-delete), notification tests (per-workspace
scoping for a user in two workspaces, read/read-all, ownership
checks), roles/permissions tests (every owner/admin/member permission
boundary above, last-owner protection on both demote and remove, and
project deletion's cascade), and full end-to-end integration tests
that spin up a *real* `WebSocketServer` on a real socket and a
hand-rolled test client (`tests/ws_test_client.py`, itself
dependency-free) to prove REST writes really push live events to
subscribed sockets -- task events, comment events, and the per-user
notification push, including multiple simultaneous subscribers, a
dead-socket cleanup case, and proof that one user's notification
channel never leaks to another connected user.

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
- **Last-owner protection is check-then-act, not atomic.** Two
  simultaneous requests to demote/remove the last two owners of a
  workspace down to zero could both pass `count_owners() <= 1` before
  either commits, in principle leaving zero owners. `run.py` runs the
  REST API single-threaded (`threaded=False`), so this can't actually
  happen with the code as shipped -- same reasoning as `next_position`'s
  read-then-insert race noted in Milestone 1's REVIEW.md. Worth a real
  lock or a `CHECK`-backed constraint if this ever moves to a threaded
  or multi-process deployment.
- **Schema changes to an *existing* `dev_collab.db` file don't
  retroactively apply.** `init_db()` uses `CREATE TABLE IF NOT EXISTS`,
  so a `dev_collab.db` left over from running Milestone 1's code will
  correctly gain the new `comments`/`notifications` tables on first
  Milestone-2 run (they didn't exist before), but the `tasks.project_id`
  foreign key's `ON DELETE CASCADE` -- added in Milestone 2 -- won't
  retroactively apply to that already-existing `tasks` table, since
  SQLite doesn't support adding/modifying a foreign key via `ALTER
  TABLE`. Deleting a project against a carried-over Milestone-1 db file
  would then fail with a foreign-key-constraint error on any tasks it
  already had, instead of cascading. Delete any pre-Milestone-2
  `dev_collab.db` (or just pick a fresh `DB_PATH`) before running this
  version -- there's no real persisted data to migrate at this stage,
  so a small real migration system wasn't worth building for it.
