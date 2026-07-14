from .conftest import auth_header, signup


def test_seed_creates_three_accounts_with_transactions(client):
    token, _ = signup(client)
    h = auth_header(token)

    r = client.post("/api/demo/seed", json={"months": 3}, headers=h)
    assert r.status_code == 201
    body = r.get_json()
    assert len(body["accounts"]) == 3
    assert body["transactions_created"] > 0

    types = {a["account_type"] for a in body["accounts"]}
    assert types == {"checking", "savings", "credit_card"}
    for a in body["accounts"]:
        assert a["is_mock"] == 1


def test_seed_transactions_are_categorized(client):
    token, _ = signup(client)
    h = auth_header(token)
    client.post("/api/demo/seed", json={"months": 3}, headers=h)

    r = client.get("/api/transactions", headers=h)
    txns = r.get_json()["transactions"]
    assert len(txns) > 0
    for t in txns:
        assert t["category"] != ""


def test_seed_is_deterministic_per_user(client):
    token, _ = signup(client, "det@example.com")
    h = auth_header(token)

    r1 = client.post("/api/demo/seed", json={"months": 3, "reset": True}, headers=h)
    txns1 = client.get("/api/transactions", headers=h).get_json()["transactions"]

    r2 = client.post("/api/demo/seed", json={"months": 3, "reset": True}, headers=h)
    txns2 = client.get("/api/transactions", headers=h).get_json()["transactions"]

    assert r1.get_json()["transactions_created"] == r2.get_json()["transactions_created"]
    amounts1 = sorted(t["amount"] for t in txns1)
    amounts2 = sorted(t["amount"] for t in txns2)
    assert amounts1 == amounts2


def test_seed_reset_removes_prior_mock_accounts(client):
    token, _ = signup(client)
    h = auth_header(token)

    client.post("/api/demo/seed", json={"months": 1}, headers=h)
    r = client.get("/api/accounts", headers=h)
    assert len(r.get_json()["accounts"]) == 3

    client.post("/api/demo/seed", json={"months": 1, "reset": True}, headers=h)
    r = client.get("/api/accounts", headers=h)
    assert len(r.get_json()["accounts"]) == 3  # replaced, not doubled


def test_seed_without_reset_adds_alongside_manual_accounts(client):
    token, _ = signup(client)
    h = auth_header(token)

    client.post("/api/accounts", json={"name": "Manual", "account_type": "checking"}, headers=h)
    client.post("/api/demo/seed", json={"months": 1}, headers=h)

    r = client.get("/api/accounts", headers=h)
    assert len(r.get_json()["accounts"]) == 4  # 1 manual + 3 mock


def test_seed_rejects_invalid_months(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/demo/seed", json={"months": 0}, headers=h)
    assert r.status_code == 400
    r = client.post("/api/demo/seed", json={"months": 13}, headers=h)
    assert r.status_code == 400


def test_recategorize_reports_checked_and_updated_counts(client):
    token, _ = signup(client)
    h = auth_header(token)
    client.post("/api/demo/seed", json={"months": 1}, headers=h)

    r = client.post("/api/demo/recategorize", headers=h)
    assert r.status_code == 200
    body = r.get_json()
    assert body["checked"] > 0
    assert body["updated"] == 0  # nothing changed, engine is deterministic
