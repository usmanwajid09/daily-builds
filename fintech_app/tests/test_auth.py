from .conftest import auth_header, signup


def test_signup_creates_user_and_returns_token(client):
    r = client.post("/api/signup", json={"email": "a@example.com", "password": "password123"})
    assert r.status_code == 201
    body = r.get_json()
    assert body["user"]["email"] == "a@example.com"
    assert "token" in body


def test_signup_rejects_invalid_email(client):
    r = client.post("/api/signup", json={"email": "not-an-email", "password": "password123"})
    assert r.status_code == 400


def test_signup_rejects_short_password(client):
    r = client.post("/api/signup", json={"email": "a@example.com", "password": "short"})
    assert r.status_code == 400


def test_signup_duplicate_email_is_409_not_500(client):
    signup(client, "dup@example.com")
    r = client.post("/api/signup", json={"email": "dup@example.com", "password": "password123"})
    assert r.status_code == 409


def test_login_success(client):
    signup(client, "b@example.com", "password123")
    r = client.post("/api/login", json={"email": "b@example.com", "password": "password123"})
    assert r.status_code == 200
    assert "token" in r.get_json()


def test_login_wrong_password(client):
    signup(client, "c@example.com", "password123")
    r = client.post("/api/login", json={"email": "c@example.com", "password": "wrongpass"})
    assert r.status_code == 401


def test_login_unknown_email(client):
    r = client.post("/api/login", json={"email": "nobody@example.com", "password": "password123"})
    assert r.status_code == 401


def test_protected_route_requires_bearer_token(client):
    r = client.get("/api/accounts")
    assert r.status_code == 401


def test_protected_route_rejects_garbage_token(client):
    r = client.get("/api/accounts", headers=auth_header("not-a-real-token"))
    assert r.status_code == 401
