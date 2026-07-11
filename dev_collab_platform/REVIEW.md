# dev-collab-platform -- self-review log

## Milestone 1 (2026-07-09)

Full diff read against `main` before writing this. Ran the test suite
(81 tests), a `pyflakes` pass, and two manual smoke tests: booting
`run.py` for real and hitting it with `curl`, then a second smoke test
that drives the actual WebSocket wire protocol end to end (raw socket
handshake -> auth -> subscribe -> REST task create -> live broadcast
received) against the running process, not just pytest's in-process
fixtures.

### Real bugs found and fixed

1. **Concurrent-write race on the same client socket (real, would have
   caused sporadic frame corruption under load).** The WS server's
   per-connection thread replies to `auth`/`subscribe` messages by
   writing to the client's socket. `Broadcaster.broadcast` -- called
   from whichever REST request thread just committed a task change --
   also writes to that same socket, from a *different* thread. Two
   threads calling `sock.sendall()` on one socket with no
   synchronization is not safe: the kernel can interleave the two
   `write()` syscalls and corrupt the WebSocket frame stream on the
   wire, even though each individual `sendall()` call looks atomic in
   Python. Fixed by wrapping every accepted socket in a
   `ClientConnection` object holding a `threading.Lock`, and routing
   *all* writes to a connection (handshake reply aside, which happens
   before any other writer exists) through `ClientConnection.send()`,
   which serializes on that lock. First attempt at this fix tried to
   just set `sock.write_lock = threading.Lock()` directly on the raw
   `socket.socket` -- that fails at runtime
   (`AttributeError: 'socket' object has no attribute 'write_lock'`,
   caught immediately by the existing test suite going from 81 passing
   to a wall of `TimeoutError`s) because CPython's socket type has no
   per-instance `__dict__`. The `ClientConnection` wrapper class is the
   actual fix.

2. **Non-compliant close handshake.** When a client sent a WebSocket
   close frame (opcode `0x8`), the server raised `ConnectionClosed` and
   closed the socket immediately without ever echoing a close frame
   back, which RFC 6455 requires for a clean close. Fixed: the server
   now sends its own close frame (echoing the client's close payload)
   before tearing the connection down, wrapped in
   `try/except OSError` in case the peer is already gone.

3. **`bool` silently satisfying `isinstance(x, int)` checks.** Python's
   `bool` is a subclass of `int`, so `isinstance(True, int)` is `True`.
   Two places validated a field as "must be an int" without excluding
   booleans: the REST `PATCH /api/tasks/<id>` route's `position` field,
   and the WebSocket `subscribe` message's `project_id` field. Either
   one would have silently accepted `{"position": true}` (storing
   `position=1`) or `{"project_id": false}` (attempting to look up
   project id `0`) instead of returning a 400/error as intended. Fixed
   both checks to `isinstance(x, int) and not isinstance(x, bool)`, and
   added regression tests for both
   (`test_update_task_rejects_boolean_position`,
   `test_subscribe_rejects_boolean_project_id`).

### Lint pass

`pyflakes app/ run.py tests/` initially flagged three real issues, all
fixed:
- `workspace_routes.py` imported `auth` but never used it (leftover
  from an earlier draft that validated email format there before that
  logic moved to `auth_routes.py`).
- `tests/test_auth_unit.py` had an unused `import time` left over from
  an earlier draft of the expiry test that used `time.sleep` before
  switching to a negative `ttl_minutes` instead.
- `tests/test_db.py`'s `test_transaction_rolls_back_on_exception`
  assigned the workspace id to `wid` and never read it.

### Reviewed and judged not a bug

- **`next_position`'s read-then-insert isn't wrapped in the same lock
  as the write.** Two concurrent `POST .../tasks` calls computing the
  same "next position" in the same column is a real race in general,
  but `run.py` boots the Flask side with `threaded=False`, so the REST
  API only ever has one request in flight at a time in this milestone
  -- there is no concurrent writer to race with. Documented as a known
  limitation in the README rather than adding locking that nothing
  exercises yet; worth revisiting if a later milestone moves to a
  threaded/gunicorn deployment.
- **One thread per WebSocket connection** and **no fragmented-frame
  reassembly** are deliberate scope decisions for a demo-scale app, not
  bugs -- both called out explicitly in the README's "Known
  limitations" section rather than left implicit.

## Milestone 2 (2026-07-11)

Full diff read against `main` before writing this. Ran the full test
suite (132 tests, up from 82), a `pyflakes` pass, and a real end-to-end
smoke test against a running `run.py` process with `curl` -- signup,
invite, comment with a mention, fetch the mentioned user's
notifications, PATCH a role, then DELETE the project and confirm the
cascade didn't raise a foreign-key error.

### Real bug found and fixed

**`PATCH /api/workspace/members/<id>` sent a notification even when
the role didn't actually change.** Re-submitting the same role (a
client retry, or a UI re-confirming the current value) unconditionally
created a `role_changed` notification reading "your role was changed
to admin" for a user whose role was already `admin` -- true but
misleading, since nothing changed. Fixed by short-circuiting to a
plain 200 with no DB write and no notification when `new_role ==
membership["role"]`. Added
`test_setting_the_same_role_is_a_noop_and_does_not_notify` as a
regression test (it initially failed against the pre-fix code, showing
`unread_count == 1` instead of the expected `0`, confirming the test
actually exercises the bug rather than passing vacuously).

### Reviewed and judged not a bug

- **Mention matching is case-insensitive by construction, not by
  accident** -- `extract_mentioned_emails` lower-cases every match, and
  `auth_routes.signup`/`login` already lower-case stored emails, so
  `@Alice@Example.COM` correctly resolves to a member stored as
  `alice@example.com`. Added
  `test_mention_matching_is_case_insensitive` as an explicit end-to-end
  regression test rather than relying on `test_mentions.py`'s unit
  tests (which only prove the *extraction* is case-folded, not that
  the full mention -> notification pipeline actually matches a real
  member) to keep this from silently regressing if either side's
  case-handling ever changes independently.
- **A user's notification WebSocket channel is keyed by `user_id`
  alone, not `(user_id, workspace_id)`.** So a person who's a member of
  two workspaces gets both workspaces' notifications on one connection
  regardless of which workspace-scoped token authenticated it. This
  was a deliberate design call, not an oversight -- confirmed correct
  by `test_notification_channel_is_per_user_not_broadcast_to_everyone`
  (isolation between *different* users) plus manual reasoning about
  what a real notification bell should do for one person across
  multiple orgs -- and called out explicitly in the README so it reads
  as intentional rather than something to "fix" later.
- **Last-owner protection (`count_owners() <= 1`) is check-then-act,
  not wrapped in a single atomic statement.** Same category of issue
  as `next_position`'s read-then-insert race flagged in Milestone 1's
  section above: a real race in general, but `run.py` runs the REST
  API single-threaded, so it can't actually be hit by the code as
  shipped. Documented, not fixed, in the README's limitations section.
- **Schema migration gap for an existing `dev_collab.db`.** `CREATE
  TABLE IF NOT EXISTS` means Milestone 2's new tables (`comments`,
  `notifications`) get created fine against a Milestone-1 leftover db
  file, but `tasks.project_id`'s new `ON DELETE CASCADE` doesn't
  retroactively apply to that table since SQLite can't add/modify a
  foreign key via `ALTER TABLE`. Deleting a project against a
  carried-over old db file would fail on existing tasks instead of
  cascading. Not fixed with a migration system (there's no real
  persisted data at this stage to migrate) -- documented in the README
  telling anyone running this locally to start from a fresh `DB_PATH`.

### Lint pass

`pyflakes app/ run.py tests/` was clean on the first pass this time --
no unused imports or variables to fix, unlike Milestone 1.
