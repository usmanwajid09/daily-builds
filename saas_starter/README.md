# saas-starter

A minimal, real multi-tenant SaaS backend: signup/login with JWTs, a
tenant + user + membership data model, plan-tier billing (Stripe TEST
mode only, never real payments), a tenant-scoped dashboard API, a
landing page, and a Docker deploy config. This is the full 2-milestone
`saas-starter` arc (Milestone 1: auth + data model; Milestone 2:
billing + gating + landing page + Docker).

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

59 tests total. Milestone 1 (29): password hashing/verification edge
cases, JWT issue/decode (including expiry and wrong-secret rejection),
the SQLite data layer, and full HTTP-level flows — signup, login
(including the multi-tenant-login disambiguation path), invite, role
enforcement (member can't invite), and an explicit cross-tenant
isolation test. Milestone 2 (+30): billing provider selection (live-key
and missing-webhook-secret guards), the fake checkout+webhook signature
roundtrip (including rejecting bad signatures and malformed payloads),
the full HTTP billing flow (checkout -> pending -> webhook/simulate ->
active), plan-tier member-limit gating on `/invite` and its lift on
upgrade, cross-tenant checkout-session isolation (tenant B can't
activate tenant A's pending session), the landing page, and two
regression tests from self-review (malformed webhook metadata failing
closed instead of 500).

## Billing (Milestone 2) — Stripe TEST mode only, never real payments

`app/billing.py` defines the plan catalogue (`PLANS`: `free` and `pro`,
each with a member/project limit) and a pluggable `BillingProvider`:

- **`FakeStripeProvider`** — a fully offline simulation of Stripe
  Checkout Sessions and webhooks (HMAC-signed local events). This is
  the provider actually exercised by every test in this repo and by
  default in any environment, since this project has no real Stripe
  account of any kind (not even test).
- **`StripeTestProvider`** — a thin wrapper around the real `stripe`
  SDK, implemented for when real Stripe TEST credentials exist, but
  never exercised here. `get_billing_provider()` only switches to it if
  `STRIPE_SECRET_KEY` is set, **and refuses to start at all if that key
  isn't a test key** (`sk_test_...`) — a live key (`sk_live_...`) is
  rejected outright. This is a hard code-level guard, not just a naming
  convention, per this arc's non-negotiable safety rule.

Billing endpoints:

- `GET /api/billing/plans` — public pricing catalogue (also drives the
  landing page's pricing table, so they can't drift out of sync).
- `GET /api/billing/plan` — the caller's tenant's current plan, status,
  limits, and usage (`@jwt_required`).
- `POST /api/billing/checkout` — **owner/admin only**. Starts an
  upgrade; returns a checkout URL (a real Stripe-hosted URL in
  `StripeTestProvider` mode, a local placeholder path in
  `FakeStripeProvider` mode, since this starter has no hosted frontend
  to redirect to). Marks the tenant's subscription `pending`.
- `POST /api/billing/webhook` — no `@jwt_required` (Stripe calls this,
  not a logged-in user); verifies the provider's signature scheme and,
  on `checkout.session.completed`, activates the tenant's plan. Rejects
  bad signatures and ignores events for unknown/already-consumed
  session ids rather than trusting metadata blindly.
- `POST /api/billing/simulate-payment` — **dev/test only**, and only
  available when running `FakeStripeProvider`; completes a pending
  checkout immediately by driving the exact same webhook-handling code
  a real event would, without needing an actual webhook round-trip.

**Plan-tier feature gating**: the free plan is capped at 3 team members.
`POST /api/auth/invite` enforces this — returns `402 Payment Required`
with `upgrade_required: true` once the cap is hit. Upgrading (via
`/checkout` + `/simulate-payment`, or a real Stripe webhook) lifts the
limit immediately.

## Landing page

`GET /` renders a server-rendered landing page (`app/templates/landing.html`)
with pricing pulled live from the same `PLANS` catalogue the API
enforces, plus a working (not mocked) signup form that calls
`/api/auth/signup` via `fetch()`.

## Docker deploy

```bash
cd saas_starter
docker compose up --build
# -> http://localhost:5000
```

`Dockerfile` runs as a non-root user, includes a container `HEALTHCHECK`
against `/api/health`, and never bakes secrets into the image — they're
read from the environment at container start. `docker-compose.yml` mounts
a named volume for the SQLite file so data survives container restarts,
and defaults `SAAS_JWT_SECRET`/Stripe keys the same way the app itself
does if left unset. **Note:** this sandbox has no Docker daemon, so the
image could only be reviewed statically here (Dockerfile conventions,
`docker-compose.yml` YAML validity) — the one thing actually verified
was that the exact `gunicorn ... "app:create_app()"` command in the
`CMD` boots correctly and serves `/api/health` and `/`, run directly
(not inside a container) in this environment.

## What's deliberately out of scope

- Real Stripe test-mode execution end-to-end (no Stripe account exists
  in this environment — see the Billing section above).
- Real email delivery for invites (temp password returned directly instead).
- Password reset / email verification flows.
- Refresh tokens (access tokens just expire after 60 min and need a new login).
- Multiple paid tiers beyond `free`/`pro`, annual billing, proration,
  cancellation flow, usage-based billing.
- Actually building/running the Docker image (no Docker daemon here —
  see Docker deploy section).

## Known limitations

- `/api/auth/login` requires `tenant_slug` when an account has multiple
  memberships. This is intentionally explicit rather than silently
  defaulting to "most recent tenant" — but it does mean a multi-tenant
  user's client needs to handle the `409` + tenant list response and
  prompt for a choice (or remember the last-used slug), which a real
  frontend for this starter would need to implement.
- Downgrading from `pro` back to `free` has no dedicated endpoint yet;
  a tenant that's already over the free plan's member limit when
  downgraded would just be blocked from *inviting further* members
  (existing members are never removed automatically) — not implemented
  or tested this milestone, since there's no downgrade path to trigger it.
