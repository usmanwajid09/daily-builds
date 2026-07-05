"""Billing routes: checkout, webhook, plan info -- Stripe TEST mode only.

See app/billing.py's module docstring for the safety rule this arc
follows. In this sandbox (no Stripe account, not even a test one), every
route here is actually exercised against FakeStripeProvider, an offline
simulation. If STRIPE_SECRET_KEY (a real "sk_test_..." key) is ever set,
these same routes transparently use StripeTestProvider instead -- the
route code doesn't change, only which BillingProvider create_app() built.
"""
import sqlite3

from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..billing import BillingError, FakeStripeProvider, PLANS, plan_limits
from ..decorators import jwt_required, role_required

bp = Blueprint("billing", __name__, url_prefix="/api/billing")


@bp.get("/plans")
def list_plans():
    """Public: the pricing catalogue (also used by the landing page)."""
    return jsonify(plans=PLANS)


@bp.get("/plan")
@jwt_required
def current_plan():
    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        subscription = db.get_subscription(conn, g.tenant_id)
        members = db.list_members_for_tenant(conn, g.tenant_id)

    plan = subscription["plan"] if subscription else "free"
    status = subscription["status"] if subscription else "active"
    limits = plan_limits(plan)
    return jsonify(
        plan=plan,
        status=status,
        limits=limits,
        usage={"members": len(members)},
    )


@bp.post("/checkout")
@jwt_required
@role_required("owner", "admin")
def create_checkout():
    """Start an upgrade to a paid plan. Returns a checkout URL the caller
    would redirect the browser to (a real Stripe-hosted page in
    production; a local placeholder path when running against
    FakeStripeProvider, since this starter has no hosted frontend).
    """
    body = request.get_json(silent=True) or {}
    plan = (body.get("plan") or "").strip()

    if plan not in PLANS:
        return jsonify(error=f"unknown plan {plan!r}"), 400
    if plan == "free":
        return jsonify(error="cannot 'checkout' into the free plan -- it has no charge"), 400

    provider = current_app.config["BILLING_PROVIDER"]
    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        tenant = db.get_tenant(conn, g.tenant_id)
        session = provider.create_checkout_session(tenant["slug"], g.tenant_id, plan)
        db.set_subscription_pending(conn, g.tenant_id, plan, session.session_id)

    return jsonify(checkout_url=session.checkout_url, session_id=session.session_id), 201


def _apply_completed_checkout(conn: sqlite3.Connection, event: dict) -> tuple[bool, str]:
    """Shared logic between the real webhook route and the dev-only
    simulate-payment helper, so both drive the exact same activation code
    path rather than the dev helper reimplementing it separately.

    Returns (applied, message).
    """
    if event.get("type") != "checkout.session.completed":
        return False, f"ignored event type {event.get('type')!r}"

    obj = event.get("data", {}).get("object", {})
    session_id = obj.get("id")
    metadata = obj.get("metadata", {}) or {}
    customer_id = obj.get("customer")

    if not session_id or "tenant_id" not in metadata or "plan" not in metadata:
        return False, "event payload missing session id or tenant/plan metadata"

    try:
        tenant_id = int(metadata["tenant_id"])
    except (TypeError, ValueError):
        # Malformed metadata (non-numeric tenant_id) -- fail closed as an
        # ignored event, not an unhandled 500. Not attacker-reachable here
        # (the signature is verified before this function ever runs, and
        # we control what goes into metadata in create_checkout_session),
        # but a corrupted/truncated payload should never crash the request.
        return False, "event metadata.tenant_id is not a valid integer"

    if metadata["plan"] not in PLANS:
        return False, f"event metadata.plan {metadata['plan']!r} is not a known plan"

    # Only activate a subscription we actually have a matching *pending*
    # checkout session for -- guards against a replayed/forged event for a
    # session_id that was never issued by our own /checkout route.
    subscription = db.get_subscription_by_checkout_session(conn, session_id)
    if subscription is None:
        return False, f"no pending subscription found for session {session_id!r}"

    if subscription["tenant_id"] != tenant_id:
        return False, "event tenant_id does not match the pending subscription's tenant"

    db.activate_subscription(conn, tenant_id, metadata["plan"], stripe_customer_id=customer_id)
    return True, f"activated {metadata['plan']} plan for tenant {tenant_id}"


@bp.post("/webhook")
def webhook():
    """Receives billing events. No @jwt_required -- this is called by
    Stripe (or, here, by whatever simulates Stripe), not by a logged-in
    user, so it authenticates via the provider's own signature scheme
    instead of a bearer token.
    """
    provider = current_app.config["BILLING_PROVIDER"]
    payload = request.get_data()
    signature = request.headers.get(provider.SIGNATURE_HEADER, "")

    try:
        event = provider.verify_and_parse_webhook(payload, signature)
    except BillingError as exc:
        return jsonify(error=str(exc)), 400

    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        applied, message = _apply_completed_checkout(conn, event)

    return jsonify(received=True, applied=applied, message=message), 200


@bp.post("/simulate-payment")
@jwt_required
@role_required("owner", "admin")
def simulate_payment():
    """Dev/test-only: completes a pending checkout session immediately,
    without needing an actual Stripe (or fake-Stripe) webhook round-trip.
    Only available when running against FakeStripeProvider -- refuses to
    do anything if a real StripeTestProvider is configured, since in that
    case the *real* webhook is the only legitimate way to confirm payment.
    """
    provider = current_app.config["BILLING_PROVIDER"]
    if not isinstance(provider, FakeStripeProvider):
        return jsonify(error="simulate-payment is only available with FakeStripeProvider"), 400

    body = request.get_json(silent=True) or {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        return jsonify(error="session_id is required"), 400

    db_path = current_app.config["DB_PATH"]
    with db.connect(db_path) as conn:
        subscription = db.get_subscription_by_checkout_session(conn, session_id)
        if subscription is None or subscription["tenant_id"] != g.tenant_id:
            return jsonify(error="no pending checkout session with that id for this tenant"), 404

        payload = provider.build_completed_event(session_id, g.tenant_id, subscription["plan"])
        event = provider.verify_and_parse_webhook(payload, provider.sign_payload(payload))
        applied, message = _apply_completed_checkout(conn, event)

    return jsonify(applied=applied, message=message), 200
