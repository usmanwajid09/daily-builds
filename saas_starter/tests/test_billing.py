"""Milestone 2: billing provider unit tests + full HTTP billing flows."""
import pytest

from app.billing import (
    BillingError,
    FakeStripeProvider,
    PLANS,
    get_billing_provider,
    plan_limits,
)


def signup(client, org_name="Acme", email="owner@acme.com", password="hunter2pass"):
    return client.post("/api/auth/signup", json={
        "org_name": org_name, "email": email, "password": password,
    })


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


# --- billing.py unit tests ---------------------------------------------------

def test_plan_limits_known_plans():
    assert plan_limits("free")["max_members"] == 3
    assert plan_limits("pro")["max_members"] == 25


def test_plan_limits_unknown_plan_raises():
    with pytest.raises(BillingError):
        plan_limits("enterprise-deluxe")


def test_get_billing_provider_defaults_to_fake():
    provider = get_billing_provider({})
    assert isinstance(provider, FakeStripeProvider)


def test_get_billing_provider_rejects_live_key():
    with pytest.raises(BillingError, match="TEST secret key"):
        get_billing_provider({"STRIPE_SECRET_KEY": "sk_live_abc123"})


def test_get_billing_provider_rejects_garbage_key():
    with pytest.raises(BillingError):
        get_billing_provider({"STRIPE_SECRET_KEY": "not-a-stripe-key-at-all"})


def test_get_billing_provider_test_key_requires_webhook_secret():
    with pytest.raises(BillingError, match="STRIPE_WEBHOOK_SECRET"):
        get_billing_provider({"STRIPE_SECRET_KEY": "sk_test_abc123"})


def test_fake_provider_checkout_session_shape():
    provider = FakeStripeProvider()
    session = provider.create_checkout_session("acme", 1, "pro")
    assert session.session_id.startswith("fake_cs_")
    assert "acme" in session.checkout_url
    assert "pro" in session.checkout_url


def test_fake_provider_checkout_rejects_unknown_plan():
    provider = FakeStripeProvider()
    with pytest.raises(BillingError):
        provider.create_checkout_session("acme", 1, "enterprise-deluxe")


def test_fake_provider_webhook_roundtrip():
    provider = FakeStripeProvider()
    payload = provider.build_completed_event("fake_cs_1", 1, "pro")
    sig = provider.sign_payload(payload)
    event = provider.verify_and_parse_webhook(payload, sig)
    assert event["type"] == "checkout.session.completed"
    assert event["data"]["object"]["metadata"]["plan"] == "pro"


def test_fake_provider_webhook_rejects_bad_signature():
    provider = FakeStripeProvider()
    payload = provider.build_completed_event("fake_cs_1", 1, "pro")
    with pytest.raises(BillingError):
        provider.verify_and_parse_webhook(payload, "not-the-right-signature")


def test_fake_provider_webhook_rejects_missing_signature():
    provider = FakeStripeProvider()
    payload = provider.build_completed_event("fake_cs_1", 1, "pro")
    with pytest.raises(BillingError):
        provider.verify_and_parse_webhook(payload, "")


def test_fake_provider_webhook_rejects_malformed_payload():
    provider = FakeStripeProvider()
    garbage = b"not json"
    sig = provider.sign_payload(garbage)
    with pytest.raises(BillingError):
        provider.verify_and_parse_webhook(garbage, sig)


# --- HTTP-level billing flows -------------------------------------------------

def test_list_plans_is_public(client):
    resp = client.get("/api/billing/plans")
    assert resp.status_code == 200
    assert set(resp.get_json()["plans"]) == {"free", "pro"}


def test_signup_starts_on_free_plan(client):
    resp = signup(client)
    token = resp.get_json()["token"]
    plan_resp = client.get("/api/billing/plan", headers=auth_header(token))
    assert plan_resp.status_code == 200
    body = plan_resp.get_json()
    assert body["plan"] == "free"
    assert body["status"] == "active"
    assert body["usage"]["members"] == 1


def test_plan_requires_auth(client):
    resp = client.get("/api/billing/plan")
    assert resp.status_code == 401


def test_checkout_rejects_unknown_plan(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/billing/checkout", json={"plan": "enterprise-deluxe"}, headers=auth_header(token))
    assert resp.status_code == 400


def test_checkout_rejects_free_plan(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/billing/checkout", json={"plan": "free"}, headers=auth_header(token))
    assert resp.status_code == 400


def _invite_temp_password(client, owner_token, email):
    resp = client.post("/api/auth/invite", json={"email": email, "role": "member"},
                        headers=auth_header(owner_token))
    return resp.get_json()["temp_password"]


def test_member_cannot_checkout(client):
    owner_token = signup(client, email="owner@acme.com").get_json()["token"]
    temp_password = _invite_temp_password(client, owner_token, "member@acme.com")
    login_resp = client.post("/api/auth/login", json={"email": "member@acme.com", "password": temp_password})
    member_token = login_resp.get_json()["token"]

    resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=auth_header(member_token))
    assert resp.status_code == 403


def test_checkout_creates_pending_subscription(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=auth_header(token))
    assert resp.status_code == 201
    assert resp.get_json()["session_id"].startswith("fake_cs_")

    plan_resp = client.get("/api/billing/plan", headers=auth_header(token))
    body = plan_resp.get_json()
    assert body["plan"] == "pro"
    assert body["status"] == "pending", "plan should not be active until the webhook confirms payment"


def test_simulate_payment_activates_plan(client):
    token = signup(client).get_json()["token"]
    checkout_resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=auth_header(token))
    session_id = checkout_resp.get_json()["session_id"]

    sim_resp = client.post("/api/billing/simulate-payment", json={"session_id": session_id},
                            headers=auth_header(token))
    assert sim_resp.status_code == 200
    assert sim_resp.get_json()["applied"] is True

    plan_resp = client.get("/api/billing/plan", headers=auth_header(token))
    body = plan_resp.get_json()
    assert body["plan"] == "pro"
    assert body["status"] == "active"


def test_simulate_payment_rejects_unknown_session(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/billing/simulate-payment", json={"session_id": "fake_cs_doesnotexist"},
                        headers=auth_header(token))
    assert resp.status_code == 404


def test_simulate_payment_cannot_activate_another_tenants_session(client):
    token_a = signup(client, org_name="Acme", email="a@acme.com").get_json()["token"]
    token_b = signup(client, org_name="Beta", email="b@beta.com").get_json()["token"]

    checkout_resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=auth_header(token_a))
    session_id = checkout_resp.get_json()["session_id"]

    # Tenant B must not be able to activate tenant A's pending session.
    resp = client.post("/api/billing/simulate-payment", json={"session_id": session_id},
                        headers=auth_header(token_b))
    assert resp.status_code == 404


def test_webhook_activates_subscription_with_valid_signature(client, app):
    token = signup(client).get_json()["token"]
    checkout_resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=auth_header(token))
    session_id = checkout_resp.get_json()["session_id"]

    provider = app.config["BILLING_PROVIDER"]
    payload = provider.build_completed_event(session_id, 1, "pro")
    sig = provider.sign_payload(payload)

    resp = client.post("/api/billing/webhook", data=payload,
                        headers={"X-Fake-Signature": sig, "Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.get_json()["applied"] is True

    plan_resp = client.get("/api/billing/plan", headers=auth_header(token))
    assert plan_resp.get_json()["plan"] == "pro"
    assert plan_resp.get_json()["status"] == "active"


def test_webhook_rejects_bad_signature(client, app):
    token = signup(client).get_json()["token"]
    checkout_resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=auth_header(token))
    session_id = checkout_resp.get_json()["session_id"]

    provider = app.config["BILLING_PROVIDER"]
    payload = provider.build_completed_event(session_id, 1, "pro")

    resp = client.post("/api/billing/webhook", data=payload,
                        headers={"X-Fake-Signature": "totally-wrong", "Content-Type": "application/json"})
    assert resp.status_code == 400

    # And the subscription must still be pending, not silently activated.
    plan_resp = client.get("/api/billing/plan", headers=auth_header(token))
    assert plan_resp.get_json()["status"] == "pending"


def test_webhook_ignores_unrelated_event_types(client, app):
    provider = app.config["BILLING_PROVIDER"]
    import json
    payload = json.dumps({"type": "customer.updated", "data": {"object": {}}}).encode("utf-8")
    sig = provider.sign_payload(payload)
    resp = client.post("/api/billing/webhook", data=payload,
                        headers={"X-Fake-Signature": sig, "Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.get_json()["applied"] is False


# --- plan-tier feature gating on /invite --------------------------------------

def test_free_plan_member_limit_enforced_on_invite(client):
    token = signup(client).get_json()["token"]
    # free plan max_members == 3; owner already counts as 1.
    r1 = client.post("/api/auth/invite", json={"email": "m1@acme.com"}, headers=auth_header(token))
    assert r1.status_code == 201
    r2 = client.post("/api/auth/invite", json={"email": "m2@acme.com"}, headers=auth_header(token))
    assert r2.status_code == 201

    r3 = client.post("/api/auth/invite", json={"email": "m3@acme.com"}, headers=auth_header(token))
    assert r3.status_code == 402
    body = r3.get_json()
    assert body["upgrade_required"] is True
    assert body["plan"] == "free"


def test_upgrading_plan_lifts_member_limit(client):
    token = signup(client).get_json()["token"]
    client.post("/api/auth/invite", json={"email": "m1@acme.com"}, headers=auth_header(token))
    client.post("/api/auth/invite", json={"email": "m2@acme.com"}, headers=auth_header(token))
    blocked = client.post("/api/auth/invite", json={"email": "m3@acme.com"}, headers=auth_header(token))
    assert blocked.status_code == 402

    checkout_resp = client.post("/api/billing/checkout", json={"plan": "pro"}, headers=auth_header(token))
    session_id = checkout_resp.get_json()["session_id"]
    client.post("/api/billing/simulate-payment", json={"session_id": session_id}, headers=auth_header(token))

    allowed = client.post("/api/auth/invite", json={"email": "m3@acme.com"}, headers=auth_header(token))
    assert allowed.status_code == 201


# --- landing page --------------------------------------------------------------

def test_landing_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"text/html" in resp.headers["Content-Type"].encode()
    assert b"Free" in resp.data
    assert b"Pro" in resp.data
    assert b"29" in resp.data


def test_webhook_rejects_non_numeric_tenant_id_gracefully(client, app):
    # Regression test: a malformed event.data.object.metadata.tenant_id
    # (non-numeric) previously crashed int(metadata["tenant_id"]) with an
    # unhandled ValueError -> 500. Should fail closed as an ignored event.
    import json
    provider = app.config["BILLING_PROVIDER"]
    event = {
        "type": "checkout.session.completed",
        "data": {"object": {"id": "fake_cs_x", "customer": "c1",
                             "metadata": {"tenant_id": "not-a-number", "plan": "pro"}}},
    }
    payload = json.dumps(event).encode("utf-8")
    sig = provider.sign_payload(payload)
    resp = client.post("/api/billing/webhook", data=payload,
                        headers={"X-Fake-Signature": sig, "Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.get_json()["applied"] is False


def test_webhook_rejects_unknown_plan_in_metadata(client, app):
    import json
    provider = app.config["BILLING_PROVIDER"]
    event = {
        "type": "checkout.session.completed",
        "data": {"object": {"id": "fake_cs_y", "customer": "c1",
                             "metadata": {"tenant_id": "1", "plan": "enterprise-deluxe"}}},
    }
    payload = json.dumps(event).encode("utf-8")
    sig = provider.sign_payload(payload)
    resp = client.post("/api/billing/webhook", data=payload,
                        headers={"X-Fake-Signature": sig, "Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.get_json()["applied"] is False
