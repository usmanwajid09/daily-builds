from .conftest import auth_header, signup


def _make_account(client, h, name="Checking", account_type="checking"):
    r = client.post("/api/accounts", json={"name": name, "account_type": account_type}, headers=h)
    return r.get_json()["id"]


def test_list_transactions_across_accounts(client):
    token, _ = signup(client)
    h = auth_header(token)
    a1 = _make_account(client, h, "Checking")
    a2 = _make_account(client, h, "Savings", "savings")

    client.post(f"/api/accounts/{a1}/transactions",
                json={"amount": -5, "merchant": "Starbucks", "posted_at": "2026-01-01"}, headers=h)
    client.post(f"/api/accounts/{a2}/transactions",
                json={"amount": 10, "posted_at": "2026-01-02"}, headers=h)

    r = client.get("/api/transactions", headers=h)
    assert r.status_code == 200
    assert len(r.get_json()["transactions"]) == 2


def test_list_transactions_filter_by_category(client):
    token, _ = signup(client)
    h = auth_header(token)
    a1 = _make_account(client, h)
    client.post(f"/api/accounts/{a1}/transactions",
                json={"amount": -5, "merchant": "Starbucks", "description": "coffee", "posted_at": "2026-01-01"},
                headers=h)
    client.post(f"/api/accounts/{a1}/transactions",
                json={"amount": -50, "merchant": "Whole Foods Market", "description": "grocery purchase",
                      "posted_at": "2026-01-02"}, headers=h)

    r = client.get("/api/transactions?category=dining", headers=h)
    txns = r.get_json()["transactions"]
    assert len(txns) == 1
    assert txns[0]["category"] == "dining"


def test_list_categories_endpoint(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.get("/api/transactions/categories", headers=h)
    assert r.status_code == 200
    assert "groceries" in r.get_json()["categories"]


def test_patch_transaction_recategorize_is_sticky(client):
    token, _ = signup(client)
    h = auth_header(token)
    a1 = _make_account(client, h)
    r = client.post(f"/api/accounts/{a1}/transactions",
                     json={"amount": -5, "merchant": "Starbucks", "description": "coffee",
                           "posted_at": "2026-01-01"}, headers=h)
    txn_id = r.get_json()["id"]
    assert r.get_json()["category"] == "dining"

    r = client.patch(f"/api/transactions/{txn_id}", json={"category": "entertainment"}, headers=h)
    assert r.status_code == 200
    assert r.get_json()["category"] == "entertainment"
    assert r.get_json()["category_is_manual"] == 1

    # recategorize should NOT touch this one since it's now manual
    client.post("/api/demo/recategorize", headers=h)
    r = client.get(f"/api/transactions/{txn_id}", headers=h)
    assert r.get_json()["category"] == "entertainment"


def test_patch_transaction_rejects_invalid_category(client):
    token, _ = signup(client)
    h = auth_header(token)
    a1 = _make_account(client, h)
    r = client.post(f"/api/accounts/{a1}/transactions",
                     json={"amount": -5, "posted_at": "2026-01-01"}, headers=h)
    txn_id = r.get_json()["id"]

    r = client.patch(f"/api/transactions/{txn_id}", json={"category": "nonsense"}, headers=h)
    assert r.status_code == 400


def test_delete_transaction(client):
    token, _ = signup(client)
    h = auth_header(token)
    a1 = _make_account(client, h)
    r = client.post(f"/api/accounts/{a1}/transactions",
                     json={"amount": -5, "posted_at": "2026-01-01"}, headers=h)
    txn_id = r.get_json()["id"]

    r = client.delete(f"/api/transactions/{txn_id}", headers=h)
    assert r.status_code == 200

    r = client.get(f"/api/transactions/{txn_id}", headers=h)
    assert r.status_code == 404


def test_cannot_access_another_users_transaction(client):
    token_a, _ = signup(client, "a2@example.com")
    token_b, _ = signup(client, "b2@example.com")
    h_a = auth_header(token_a)
    h_b = auth_header(token_b)

    a1 = _make_account(client, h_a)
    r = client.post(f"/api/accounts/{a1}/transactions",
                     json={"amount": -5, "posted_at": "2026-01-01"}, headers=h_a)
    txn_id = r.get_json()["id"]

    r = client.get(f"/api/transactions/{txn_id}", headers=h_b)
    assert r.status_code == 404
    r = client.patch(f"/api/transactions/{txn_id}", json={"category": "dining"}, headers=h_b)
    assert r.status_code == 404
    r = client.delete(f"/api/transactions/{txn_id}", headers=h_b)
    assert r.status_code == 404
