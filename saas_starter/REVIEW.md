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
