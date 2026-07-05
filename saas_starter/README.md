# saas-starter

A minimal, real multi-tenant SaaS backend: signup/login with JWTs,
a tenant + user + membership data model, and a tenant-scoped dashboard
API. This is Milestone 1 of the `saas-starter` arc. Billing (Stripe
TEST mode only) and plan-tier feature gating land in Milestone 2 — this
milestone intentionally ships **no payment code at all**.

## Why this data model

Most "multi-tenant" starters model one user account per company. This
one models a **global user identity** (`users`) joined to any number of
**tenants** through a **memberships** table with a per-tenant `role`.
That's what lets a real person belong to more than one organization —
common in practice (an agency employee working across client orgs, a
consultant, etc.) — with a distinct role in each.

```
tenants          users              memberships
--------         --------           --------------------
id               id                 id
name             email (unique)     user_id  -> users.id
slug (unique)    password_hash      tenant_id -> tenants.id
created_at       created_at         role (owner|admin|member)
                                    UNIQUE(user_id, tenant_id)
```

Every tenant-scoped query filters on `tenant_id`, and that `tenant_id`
always comes from the verified JWT — never from a client-supplied
field — so there's no request parameter to tamper with to read another
tenant's data.

## Auth flow

- `POST /api/auth/signup` — creates a **new tenant** plus its first user
  (role `owner`). Returns a JWT scoped to that tenant.
- `POST /api/auth/login` — email + password. If the account belongs to
  exactly one tenant, logs straight in. If it belongs to more than one,
  returns `409` with the list of tenant slugs and expects `tenant_slug`
  in a follow-up request to disambiguate which org context to issue the
  token for.
- `POST /api/auth/invite` — **owner/admin only**. Invites an email into
  the caller's current tenant. If that email has no account yet, one is
  created with a random temporary password returned in the response
  (dev-only stand-in for a real emailed invite link — noted in the code
  and here, not swept under the rug).
- `GET /api/auth/me` — returns the caller's identity, current tenant,
  and role, from the token.
- `GET /api/dashboard`, `GET /api/tenants/me/members` — tenant-scoped
  reads, protected by `@jwt_required`.

JWTs are HS256, signed with `SAAS_JWT_SECRET` (defaults to an obviously
fake dev value — **must** be overridden via env var for anything beyond
local testing), and expire after 60 minutes by default.

## Running it

```bash
cd saas_starter
pip install -r requirements.txt
SAAS_JWT_SECRET=some-long-random-value SAAS_DB_PATH=saas_starter.db python run.py
# server on http://localhost:5000
```

Example session:

```bash
curl -s localhost:5000/api/auth/signup -X POST -H 'Content-Type: application/json' \
  -d '{"org_name": "Acme", "email": "owner@acme.com", "password": "hunter2pass"}'
# => { "token": "...", "tenant": {"slug": "acme", ...}, "role": "owner", ... }

curl -s localhost:5000/api/dashboard -H 'Authorization: Bearer <token>'
```

## Tests

```bash
cd saas_starter
pip install -r requirements.txt
python -m pytest tests/ -v
```

28 tests: password hashing/verification edge cases, JWT issue/decode
(including expiry and wrong-secret rejection), the SQLite data layer,
and full HTTP-level flows — signup, login (including the
multi-tenant-login disambiguation path), invite, role enforcement
(member can't invite), and an explicit cross-tenant isolation test that
asserts one tenant's member list never contains another tenant's data.

## What's deliberately out of scope for this milestone

- Any payment/billing code (Stripe TEST mode, plan gating) — Milestone 2.
- Real email delivery for invites (temp password returned directly instead).
- Password reset / email verification flows.
- Refresh tokens (access tokens just expire after 60 min and need a new login).

## Known limitation

`/api/auth/login` requires `tenant_slug` when an account has multiple
memberships. This is intentionally explicit rather than silently
defaulting to "most recent tenant" — but it does mean a multi-tenant
user's client needs to handle the `409` + tenant list response and
prompt for a choice (or remember the last-used slug), which a real
frontend for this starter would need to implement.
