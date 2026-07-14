from .conftest import auth_header, signup


def test_create_and_list_account(client):
    token, _ = signup(client)
    h = auth_header(token)

    r = client.post("/api/accounts", json={"name": "My Checking", "account_type": "checking"}, headers=h)
    assert r.status_code == 201
    account = r.get_json()
    assert account["name"] == "My Checking"
    assert account["balance"] == 0

    r = client.get("/api/accounts", headers=h)
    assert r.status_code == 200
    accounts = r.get_json()["accounts"]
    assert len(accounts) == 1


def test_create_account_rejects_bad_type(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Bad", "account_type": "bitcoin_wallet"}, headers=h)
    assert r.status_code == 400


def test_create_account_requires_name(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"account_type": "checking"}, headers=h)
    assert r.status_code == 400


def test_get_account_not_found(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.get("/api/accounts/999", headers=h)
    assert r.status_code == 404


def test_cannot_access_another_users_account(client):
    token_a, _ = signup(client, "a@example.com")
    token_b, _ = signup(client, "b@example.com")

    r = client.post("/api/accounts", json={"name": "A's account", "account_type": "checking"},
                     headers=auth_header(token_a))
    account_id = r.get_json()["id"]

    # user B tries to read user A's account -- must be 404, not 403, so
    # user B can't distinguish "not mine" from "doesn't exist"
    r = client.get(f"/api/accounts/{account_id}", headers=auth_header(token_b))
    assert r.status_code == 404


def test_delete_account_cascades_transactions(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Temp", "account_type": "checking"}, headers=h)
    account_id = r.get_json()["id"]

    client.post(f"/api/accounts/{account_id}/transactions",
                json={"amount": -10, "merchant": "Test", "description": "", "posted_at": "2026-01-01"},
                headers=h)

    r = client.delete(f"/api/accounts/{account_id}", headers=h)
    assert r.status_code == 200

    r = client.get(f"/api/accounts/{account_id}/transactions", headers=h)
    assert r.status_code == 404


def test_create_transaction_auto_categorizes(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Checking", "account_type": "checking"}, headers=h)
    account_id = r.get_json()["id"]

    r = client.post(
        f"/api/accounts/{account_id}/transactions",
        json={"amount": -12.50, "merchant": "Starbucks", "description": "coffee", "posted_at": "2026-02-01"},
        headers=h,
    )
    assert r.status_code == 201
    txn = r.get_json()
    assert txn["category"] == "dining"
    assert txn["category_is_manual"] == 0


def test_create_transaction_honors_explicit_category(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Checking", "account_type": "checking"}, headers=h)
    account_id = r.get_json()["id"]

    r = client.post(
        f"/api/accounts/{account_id}/transactions",
        json={"amount": -12.50, "merchant": "Starbucks", "description": "coffee",
              "posted_at": "2026-02-01", "category": "entertainment"},
        headers=h,
    )
    assert r.status_code == 201
    txn = r.get_json()
    assert txn["category"] == "entertainment"
    assert txn["category_is_manual"] == 1


def test_create_transaction_rejects_bad_category(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Checking", "account_type": "checking"}, headers=h)
    account_id = r.get_json()["id"]

    r = client.post(
        f"/api/accounts/{account_id}/transactions",
        json={"amount": -1, "posted_at": "2026-02-01", "category": "not-a-real-category"},
        headers=h,
    )
    assert r.status_code == 400


def test_create_transaction_requires_posted_at(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Checking", "account_type": "checking"}, headers=h)
    account_id = r.get_json()["id"]

    r = client.post(
        f"/api/accounts/{account_id}/transactions",
        json={"amount": -1, "merchant": "Test"},
        headers=h,
    )
    assert r.status_code == 400


def test_account_balance_reflects_transactions(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Checking", "account_type": "checking"}, headers=h)
    account_id = r.get_json()["id"]

    client.post(f"/api/accounts/{account_id}/transactions",
                json={"amount": 100, "posted_at": "2026-01-01"}, headers=h)
    client.post(f"/api/accounts/{account_id}/transactions",
                json={"amount": -30, "posted_at": "2026-01-02"}, headers=h)

    r = client.get(f"/api/accounts/{account_id}", headers=h)
    assert r.get_json()["balance"] == 70


def test_create_account_normalizes_type_case(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Mixed Case", "account_type": "Checking"}, headers=h)
    assert r.status_code == 201
    assert r.get_json()["account_type"] == "checking"


def test_create_transaction_normalizes_category_case(client):
    token, _ = signup(client)
    h = auth_header(token)
    r = client.post("/api/accounts", json={"name": "Checking", "account_type": "checking"}, headers=h)
    account_id = r.get_json()["id"]
    r = client.post(
        f"/api/accounts/{account_id}/transactions",
        json={"amount": -5, "posted_at": "2026-01-01", "category": "Dining"},
        headers=h,
    )
    assert r.status_code == 201
    assert r.get_json()["category"] == "dining"
