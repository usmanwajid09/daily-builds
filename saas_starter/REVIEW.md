# Self-review — saas-starter (Milestone 1: auth scaffold + multi-tenant data model + dashboard skeleton)

What I checked, and what I found:

1. **Check-then-insert race on uniqueness constraints (found and fixed).**
   Both `/api/auth/signup` (unique email, unique tenant slug) and
   `/api/auth/invite` (unique `(user_id, tenant_id)` membership) queried
   for an existing row, then inserted, without an atomic
   check-and-insert. Two concurrent requests racing that window could
   both pass the check and then have the second `INSERT` raise
   `sqlite3.IntegrityError`, which the `db.connect()` context manager
   re-raises and Flask would turn into an unhandled `500`. Wrapped both
   endpoints' write blocks in `try/except sqlite3.IntegrityError` and
   return a proper `409` instead. Added
   `test_signup_race_on_duplicate_email_returns_409_not_500`, which
   monkeypatches `create_user` to insert a colliding row mid-request to
   actually exercise the race path deterministically, rather than relying
   on real concurrency (which would be flaky in a test).

2. **Tenant isolation is structural, not just tested.** Every tenant-scoped
   query (`list_members_for_tenant`, the dashboard route) takes
   `tenant_id` from `g.tenant_id`, which is only ever set by
   `jwt_required` from the *verified* token's claim — no route reads a
   tenant id from the request body/query string. Still wrote an explicit
   `test_tenant_isolation_across_two_orgs` test (two tenants, invite a
   member into one, assert the other tenant's member list never contains
   it) rather than trusting the design by inspection alone.

3. **Revoked membership doesn't get a free ride on an unexpired token
   (verified, not assumed).** `jwt_required` re-checks
   `db.get_membership(user_id, tenant_id)` on every request rather than
   trusting the JWT's `role` claim for the full 60-minute TTL. This means
   if an admin removed a teammate from a tenant, that teammate's existing
   token stops working on the very next request, not just after it
   expires. (There's no `/api/tenants/:id/members/:user_id` DELETE
   endpoint yet in this milestone, so this path isn't reachable through
   the current API surface — it's a decision made for when member removal
   ships, documented here so it isn't forgotten or re-litigated then.)

4. **`verify_password` never raises on a malformed hash.** If the stored
   `password_hash` were ever corrupted or the wrong type, `bcrypt.checkpw`
   raises `ValueError`/`TypeError`; caught explicitly and treated as "wrong
   password" (`False`) rather than letting it 500 the login endpoint.
   Covered by `test_verify_password_handles_garbage_hash_gracefully`.

5. **Password minimum length enforced server-side (8 chars), not just
   suggested.** `hash_password` raises `ValueError` below that, and
   `signup` turns that into a `400` rather than accepting a weak password
   and hashing it anyway. No maximum length or complexity rules — out of
   scope for a starter, noted rather than silently assumed to be handled.

6. **Login's multi-tenant disambiguation path was the trickiest thing to
   get right and is now explicitly tested end-to-end.**
   `test_login_disambiguates_multi_tenant_user` creates one user with
   memberships in two tenants (via invite), confirms login without
   `tenant_slug` returns `409` with both tenant slugs listed, and that
   supplying the right `tenant_slug` resolves to the correct tenant's
   token. This is the one part of the API a real frontend for this
   starter would need extra handling for (see README's "Known
   limitation").

7. **Role enforcement on `/invite` is tested for the negative case, not
   just the happy path.** `test_invite_forbidden_for_member_role` invites
   a `member`, logs in as them, and confirms *they* can't invite someone
   else (`403`) — checks the decorator order (`jwt_required` then
   `role_required`) actually composes correctly rather than assuming it.

8. **JWT secret has an obviously-fake dev default, not a silently-weak
   real-looking one.** `SAAS_JWT_SECRET` defaults to
   `"dev-secret-change-me"` so it's impossible to mistake for a
   production-ready value if someone forgets to set the env var — flagged
   again in the README's setup instructions.

9. **Nothing generated is tracked.** `saas_starter.db` / any `*.db` file
   created by `run.py` or manual testing is excluded via the repo-root
   `.gitignore` (extended this milestone to also cover `*.sqlite`,
   `*.sqlite3`, `.env`, `node_modules/`, on top of the existing Python
   cache entries). Confirmed with `git ls-files` before every commit.

Known gaps carried forward (not bugs, just not in scope for Milestone 1):
refresh tokens (access tokens simply expire after 60 min), password
reset / email verification, and of course all billing/plan-gating —
that's explicitly Milestone 2's job per `ARC_QUEUE.md`, and this
milestone ships zero payment-related code on purpose.

## Milestone 2 (2026-07-05): Stripe TEST-mode billing, plan-tier gating, landing page, Docker

What I checked, and what I found:

1. **Two unused imports (found via `pyflakes`, not just eyeballing).**
   `typing.Optional` in `billing.py` and `flask.current_app` in
   `landing_routes.py` were both imported but never used. Small, but
   running a linter as part of self-review is exactly how these get
   caught instead of accumulating -- fixed both. `pyflakes` now passes
   clean on `app/`.

2. **Malformed webhook metadata could 500 (found and fixed).**
   `_apply_completed_checkout` did `int(metadata["tenant_id"])` with no
   guard. A corrupted or malformed event payload (non-numeric
   `tenant_id`) would raise an unhandled `ValueError`, which Flask turns
   into a 500 for what should be a "yes, received, but ignored" response
   -- exactly the same category of bug as milestone 1's
   check-then-insert race (an edge case turning into an unhandled crash
   instead of a clean, expected response). Not attacker-reachable today
   (the payload's signature is verified before this code runs, and we
   control what goes into `metadata` ourselves in `create_checkout_session`),
   but "not attacker-reachable yet" isn't the same as "safe to leave
   unguarded" -- a future real Stripe integration, a bug in our own
   metadata construction, or a partially-corrupted payload could all hit
   this. Fixed with an explicit `try/except (TypeError, ValueError)`
   that fails closed as an ignored event instead of crashing. Also added
   a matching check that `metadata["plan"]` is one of the known `PLANS`
   keys, for the same reason. Two regression tests added
   (`test_webhook_rejects_non_numeric_tenant_id_gracefully`,
   `test_webhook_rejects_unknown_plan_in_metadata`).

3. **A real TOCTOU race exists on the free-plan member-limit gate --
   identified, deliberately NOT fixed this milestone, documented here
   instead.** `/api/auth/invite`'s plan-tier gate does a
   count-then-insert: read the current member count, compare to the
   plan's `max_members`, then insert if under the cap. Unlike
   milestone 1's signup/invite race (which was against a `UNIQUE`
   constraint and therefore *always* surfaces as a catchable
   `sqlite3.IntegrityError` no matter how the timing lands), this is a
   pure business-rule check with no database constraint backing it.
   Two invite requests landing concurrently, both reading the count
   just under the cap, could both insert successfully and put a tenant
   one member over its plan's limit. I considered fixing this with an
   explicit `BEGIN IMMEDIATE` to force early write-lock acquisition, but
   decided against shipping that change this milestone: it's real
   concurrency-control code I could not verify with a deterministic
   test in the time available (true concurrency races aren't reliably
   reproducible with monkeypatching the way milestone 1's
   check-then-insert race was), and shipping unverified locking logic
   risks introducing a worse bug than the one it fixes. This is a soft
   business-rule overshoot (a tenant ends up with N+1 members instead of
   N in a narrow race window), not a crash or a security hole, so
   documenting it as a known, accepted limitation and deferring a
   proper fix (most likely: move the count check into a single atomic
   SQL statement, or add a DB trigger) is the more honest call than a
   rushed, unverified fix. Noted in README's Known Limitations.

4. **Repeated `/checkout` calls orphan the previous session id (accepted
   behavior, documented, not fixed).** `set_subscription_pending`
   overwrites a tenant's single subscription row's
   `stripe_checkout_session_id` on every call. If a tenant calls
   `/checkout` twice before completing payment, the first session's id
   is no longer the one stored on the row -- a webhook arriving later
   for that first, abandoned session would find no matching pending
   subscription and be (correctly, safely) ignored rather than
   activating anything. This fails *safe* (an abandoned checkout is
   just ignored, never incorrectly activates a plan), so I judged it
   not worth adding idempotency-key infrastructure for in this
   milestone -- real Stripe integrations handle exactly this by letting
   abandoned Checkout Sessions expire naturally.

5. **Every billing safety guard was tested, not just implemented.**
   `get_billing_provider` rejects a live key (`sk_live_...`), rejects a
   key that isn't recognizably a Stripe key at all, and requires
   `STRIPE_WEBHOOK_SECRET` whenever a real test key is configured -- all
   three have dedicated tests. The webhook route's signature
   verification is tested for a valid signature, a tampered one, and a
   missing one. Cross-tenant isolation is tested at the billing layer
   too, not just the membership layer from milestone 1:
   `test_simulate_payment_cannot_activate_another_tenants_session`
   confirms tenant B cannot activate tenant A's pending checkout session
   by guessing/reusing its session id.

6. **Docker deploy config could only be reviewed statically, not built
   or run -- this sandbox has no Docker daemon (`docker` is not even
   installed).** What I could and did verify: `docker-compose.yml` is
   valid YAML (parsed with `PyYAML` and inspected structurally) with the
   expected service/volume/healthcheck shape; and, most importantly, the
   *exact* command in the Dockerfile's `CMD`
   (`gunicorn --bind 0.0.0.0:5000 --workers 2 "app:create_app()"`) was
   run directly in this sandbox (not inside a container) against the
   real app package, and it correctly served both `/api/health` and `/`.
   What I could NOT verify: that the image actually builds (dependency
   resolution inside the container, base image compatibility), that the
   non-root user/permissions setup on `/data` actually works end-to-end,
   or that `docker compose up` wires the volume and env vars together
   correctly. Flagging this honestly rather than claiming a "tested"
   Docker setup I couldn't actually run -- this is the same
   "acknowledge what's blocked/unavailable, don't paper over it" pattern
   the ai-trading-bot arc used for blocked market-data APIs.

7. **Nothing generated is tracked.** Confirmed via `git ls-files` before
   every commit this milestone; no `saas_starter.db`, `__pycache__/`, or
   `.pytest_cache/` slipped in. `.dockerignore` was added alongside the
   Dockerfile so the *image* also excludes generated/dev-only content
   (`tests/`, `.git`, `REVIEW.md`) even though the repo's own
   `.gitignore` already keeps them out of version control.

### Test coverage

59 tests total: 29 carried over from Milestone 1 plus 30 new this
milestone (28 in the initial `test_billing.py` commit, +2 regression
tests added during this self-review for the malformed-webhook-metadata
fix). New coverage: billing-provider selection and its safety guards,
the fake checkout+webhook signature roundtrip (valid, tampered, missing,
and malformed-payload signatures), the full HTTP billing flow
(checkout -> pending -> webhook-or-simulate -> active), plan-tier
member-limit gating on `/invite` and its lift on upgrade, cross-tenant
checkout-session isolation, and the landing page.

### Safety rule compliance (per ARC_QUEUE.md)

No real payment credentials exist anywhere in this codebase or this
sandbox. `StripeTestProvider` is fully implemented but never
instantiated in any test or manual run -- every exercised code path in
this milestone goes through `FakeStripeProvider`. `get_billing_provider`
hard-rejects a live Stripe key by construction, not by convention, and
that rejection itself is unit-tested
(`test_get_billing_provider_rejects_live_key`).
