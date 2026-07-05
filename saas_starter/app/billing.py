"""Plan tiers and a pluggable billing provider abstraction.

SAFETY RULE (from ARC_QUEUE.md, non-negotiable for this arc): saas-starter
must never use real payment credentials or move real money. This module
enforces that at the code level, not just by convention:

  - get_billing_provider() raises a hard error and refuses to start if
    STRIPE_SECRET_KEY is set to anything that isn't a Stripe *test* key
    (i.e. it must start with "sk_test_"). A live key ("sk_live_...")
    is rejected outright.
  - This sandbox has no Stripe account of any kind (not even test), so
    the default and actually-exercised path in this milestone is
    FakeStripeProvider: a fully offline simulation of Stripe Checkout
    Sessions and webhook events that mirrors the real API's shape
    closely enough that swapping in a genuine "sk_test_..." key later
    (via the STRIPE_SECRET_KEY env var) is a drop-in change, not a
    rewrite. StripeTestProvider (the real-Stripe-test-mode wrapper) is
    implemented for that future swap but is never exercised end-to-end
    here, since we have no test credentials to exercise it with -- this
    is the same "real API blocked/unavailable, build a documented
    from-scratch/mocked stand-in" pattern already used by the
    ai-trading-bot arc for market data.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Protocol


class BillingError(Exception):
    """Raised for any billing-configuration or webhook-verification failure."""


# --- Plan catalogue ----------------------------------------------------------

PLANS = {
    "free": {
        "label": "Free",
        "price_usd": 0,
        "max_members": 3,
        "max_projects": 1,
    },
    "pro": {
        "label": "Pro",
        "price_usd": 29,
        "max_members": 25,
        "max_projects": 20,
        # Placeholder Stripe Price ID -- would be a real "price_..." id from
        # the Stripe *test* dashboard once real test credentials exist.
        "stripe_price_id": "price_test_pro_placeholder",
    },
}

DEFAULT_PLAN = "free"


def plan_limits(plan: str) -> dict:
    if plan not in PLANS:
        raise BillingError(f"unknown plan {plan!r}")
    return PLANS[plan]


# --- Provider interface -------------------------------------------------------

@dataclass(frozen=True)
class CheckoutSession:
    session_id: str
    checkout_url: str


class BillingProvider(Protocol):
    def create_checkout_session(
        self, tenant_slug: str, tenant_id: int, plan: str
    ) -> CheckoutSession: ...

    def verify_and_parse_webhook(self, payload: bytes, signature_header: str) -> dict: ...


# --- Fake (offline) provider -- the one actually exercised in this sandbox ---

FAKE_EVENT_TYPE_COMPLETED = "checkout.session.completed"


class FakeStripeProvider:
    """Fully offline stand-in for Stripe Checkout + webhooks.

    Signature scheme: HMAC-SHA256 of the raw JSON body using a local
    "fake webhook secret", sent as a plain hex digest in the
    `X-Fake-Signature` header. This is deliberately analogous to how
    Stripe's real `Stripe-Signature` header works (HMAC over the payload)
    so the verification *shape* -- reject tampered/unsigned payloads --
    is genuinely exercised, without needing any real Stripe account.
    """

    SIGNATURE_HEADER = "X-Fake-Signature"

    def __init__(self, webhook_secret: str = "test-only-fake-webhook-secret"):
        self.webhook_secret = webhook_secret

    def create_checkout_session(
        self, tenant_slug: str, tenant_id: int, plan: str
    ) -> CheckoutSession:
        if plan not in PLANS:
            raise BillingError(f"unknown plan {plan!r}")
        session_id = f"fake_cs_{uuid.uuid4().hex[:24]}"
        # Not a real URL to click -- this is a backend starter, not a
        # hosted checkout page. A real deployment with real Stripe test
        # keys would get back an actual https://checkout.stripe.com/... URL.
        checkout_url = f"/billing/fake-checkout/{session_id}?tenant={tenant_slug}&plan={plan}"
        return CheckoutSession(session_id=session_id, checkout_url=checkout_url)

    def sign_payload(self, payload: bytes) -> str:
        return hmac.new(self.webhook_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def build_completed_event(self, session_id: str, tenant_id: int, plan: str) -> bytes:
        """Build the raw JSON payload for a simulated
        'checkout.session.completed' event -- used by tests and by the
        (dev-only) /api/billing/simulate-payment endpoint to drive the
        exact same webhook code path a real Stripe event would.
        """
        event = {
            "id": f"fake_evt_{uuid.uuid4().hex[:24]}",
            "type": FAKE_EVENT_TYPE_COMPLETED,
            "created": int(time.time()),
            "data": {
                "object": {
                    "id": session_id,
                    "customer": f"fake_cus_{tenant_id}",
                    "metadata": {"tenant_id": str(tenant_id), "plan": plan},
                }
            },
        }
        return json.dumps(event).encode("utf-8")

    def verify_and_parse_webhook(self, payload: bytes, signature_header: str) -> dict:
        if not signature_header:
            raise BillingError("missing webhook signature")
        expected = self.sign_payload(payload)
        if not hmac.compare_digest(expected, signature_header):
            raise BillingError("webhook signature verification failed")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BillingError("malformed webhook payload") from exc


# --- Real Stripe TEST-mode provider (implemented, never exercised here) -----

class StripeTestProvider:
    """Thin wrapper around the real `stripe` SDK, restricted to test mode.

    Never instantiate this with anything but an "sk_test_..." secret --
    see get_billing_provider(), which is the only place this class should
    be constructed and which enforces that restriction before this
    class's __init__ ever runs.
    """

    SIGNATURE_HEADER = "Stripe-Signature"

    def __init__(self, secret_key: str, webhook_secret: str):
        if not secret_key.startswith("sk_test_"):
            # Defense in depth: even if get_billing_provider()'s check were
            # ever bypassed, this class refuses to hold a non-test key.
            raise BillingError("StripeTestProvider requires a Stripe TEST secret key (sk_test_...)")
        import stripe  # imported lazily: only needed if real Stripe is configured

        self._stripe = stripe
        self._stripe.api_key = secret_key
        self.webhook_secret = webhook_secret

    def create_checkout_session(
        self, tenant_slug: str, tenant_id: int, plan: str
    ) -> CheckoutSession:
        price_id = plan_limits(plan).get("stripe_price_id")
        if not price_id:
            raise BillingError(f"plan {plan!r} has no Stripe price configured")
        session = self._stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"https://example.invalid/billing/success?tenant={tenant_slug}",
            cancel_url=f"https://example.invalid/billing/cancel?tenant={tenant_slug}",
            metadata={"tenant_id": str(tenant_id), "plan": plan},
        )
        return CheckoutSession(session_id=session["id"], checkout_url=session["url"])

    def verify_and_parse_webhook(self, payload: bytes, signature_header: str) -> dict:
        try:
            event = self._stripe.Webhook.construct_event(
                payload, signature_header, self.webhook_secret
            )
        except (ValueError, self._stripe.error.SignatureVerificationError) as exc:
            raise BillingError(f"webhook verification failed: {exc}") from exc
        return event


def get_billing_provider(config: dict) -> BillingProvider:
    """Select a billing provider from app config.

    Defaults to FakeStripeProvider. Only switches to the real
    StripeTestProvider if STRIPE_SECRET_KEY is explicitly set -- and even
    then, only if it's unmistakably a test-mode key. Any live key
    ("sk_live_...", or literally anything not starting with "sk_test_")
    is rejected with a BillingError rather than silently used, since a
    misconfigured env var here would otherwise be exactly how a "test"
    deployment starts moving real money by accident.
    """
    secret_key = config.get("STRIPE_SECRET_KEY")
    if not secret_key:
        return FakeStripeProvider(webhook_secret=config.get("FAKE_WEBHOOK_SECRET", "test-only-fake-webhook-secret"))

    if not secret_key.startswith("sk_test_"):
        raise BillingError(
            "STRIPE_SECRET_KEY must be a Stripe TEST secret key (starts with "
            "'sk_test_'). Live keys are refused by design -- this arc must "
            "never move real money."
        )
    webhook_secret = config.get("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        raise BillingError("STRIPE_WEBHOOK_SECRET is required when STRIPE_SECRET_KEY is set")
    return StripeTestProvider(secret_key, webhook_secret)
