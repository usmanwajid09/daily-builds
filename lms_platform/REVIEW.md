# lms-platform -- self-review log

Running log of self-review findings per milestone. Newest first.

## Milestone 1 (2026-07-11)

Full diff read against `main`, a manual end-to-end smoke test via the
Flask test client (signup -> create course -> create lesson -> enroll
-> complete -> progress -> upload), the full pytest suite, and a
`pyflakes` pass.

**Two real bugs found and fixed**, both with regression tests verified
to fail against the pre-fix code before being confirmed fixed:

1. **Duplicate `order_index` crashed with a raw 500** instead of a
   clean error. `lessons` has a `UNIQUE (course_id, order_index)`
   constraint, but `create_lesson`/`update_lesson` in
   `routes/lesson_routes.py` let the resulting `sqlite3.IntegrityError`
   propagate straight out of the route, so Flask's default exception
   handler turned it into an unhandled 500 with a stack trace instead
   of a validation-style error a client could act on. Hit via both the
   create path (two lessons created with the same explicit
   `order_index`) and the update path (PATCHing a lesson's
   `order_index` to collide with a sibling). Fixed by wrapping both
   writes in `try/except sqlite3.IntegrityError` and returning a clean
   `409` naming the conflicting `order_index`. Two regression tests
   added (`test_create_lesson_rejects_duplicate_order_index`,
   `test_update_lesson_rejects_duplicate_order_index`); both confirmed
   to fail with `500` against the pre-fix code before the fix, and
   pass (`409`) after.

2. **Signup had the exact check-then-insert race saas_starter's
   signup/invite hit and fixed in PR #7** -- and this milestone
   reintroduced the same pattern from scratch instead of learning from
   it. `POST /api/signup` checked `get_user_by_email() is not None`
   and then inserted in a separate statement; under two concurrent
   signups for the same email, both could pass the check before either
   commits, and the loser would crash with an unhandled
   `sqlite3.IntegrityError` on the `users.email` UNIQUE constraint
   instead of a normal 409. Fixed the same way saas_starter did:
   wrapped the insert in `try/except sqlite3.IntegrityError` and
   return 409 there too, so the UNIQUE constraint -- not the
   read-then-write check -- is the actual source of truth. Regression
   test (`test_signup_race_on_duplicate_email_returns_409_not_500`)
   simulates the race properly: a conflicting row is inserted directly
   (standing in for the "concurrent" request that already won), and
   `db.get_user_by_email` is monkeypatched to lie and report no
   duplicate for this request's own check -- so the only thing that
   can catch the conflict is the INSERT hitting the UNIQUE constraint
   for real. (An earlier version of this test inserted the conflicting
   row *before* patching the check function, which meant the ordinary
   pre-check caught it and the test passed even against the buggy
   pre-fix code -- a false-negative regression test that would never
   have caught a regression. Caught by deliberately running it against
   the pre-fix source and seeing it pass when it should have failed,
   then rewritten to actually exercise the race.)

**Verification method note**: for both fixes, the pre-fix source was
checked out from the milestone's earlier commit, the new regression
test was run against it standalone to confirm it failed for the right
reason (the exact assertion/traceback expected), then the fix was
restored and the test re-run to confirm it passed. This caught the
race-condition test's initial false-negative (see above) that a
"does the fixed code pass" check alone would have missed.

**Reviewed and not changed:**

- `update_lesson`'s route handler never lets a PATCH set
  `content_type` (only `title`, `content`, `order_index` are read from
  the request body into the update), so a client cannot bypass the
  create-time restriction that `content_type='file'` can only be set
  via the `/upload` endpoint. Verified by reading the handler, not just
  asserting -- there is no code path that copies `content_type` from
  the PATCH body into the update fields dict.
- The upload endpoint reads the whole file into memory before writing
  it (`upload.read()`) rather than streaming to disk. Bounded by both
  Flask's `MAX_CONTENT_LENGTH` (12 MB request cap) and an explicit
  10 MB per-file check, so this is a deliberate simplicity-over-
  streaming tradeoff at this scale, not an unbounded-memory risk --
  flagged here rather than silently left in case a future milestone
  needs to raise the size limit meaningfully, at which point streaming
  would be worth revisiting.
- `db.connect()` passes `check_same_thread=False` so the single
  process-wide connection can serve Flask's threaded dev server
  (`run.py` uses `threaded=True`). No manual locking around writes is
  added, unlike dev_collab_platform's raw-socket WebSocket server --
  this app has no long-lived background writer thread the way the WS
  broadcaster does, so ordinary SQLite serialization (SQLite itself
  serializes writes on a single connection) is sufficient here; noted
  in case a later milestone adds a background job that writes
  concurrently with request handlers.
