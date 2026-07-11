from lms_platform.tests.conftest import auth_header, signup


def test_signup_creates_user_and_token(client):
    r = client.post("/api/signup", json={"email": "a@example.com", "password": "password123", "role": "instructor"})
    assert r.status_code == 201
    body = r.get_json()
    assert body["user"]["email"] == "a@example.com"
    assert body["user"]["role"] == "instructor"
    assert body["token"]


def test_signup_rejects_bad_role(client):
    r = client.post("/api/signup", json={"email": "a@example.com", "password": "password123", "role": "admin"})
    assert r.status_code == 400


def test_signup_rejects_short_password(client):
    r = client.post("/api/signup", json={"email": "a@example.com", "password": "short", "role": "student"})
    assert r.status_code == 400


def test_signup_rejects_duplicate_email(client):
    signup(client, "dup@example.com", "password123", "student")
    r = client.post("/api/signup", json={"email": "dup@example.com", "password": "password123", "role": "student"})
    assert r.status_code == 409


def test_signup_rejects_missing_email(client):
    r = client.post("/api/signup", json={"password": "password123", "role": "student"})
    assert r.status_code == 400


def test_login_success(client):
    signup(client, "b@example.com", "password123", "student")
    r = client.post("/api/login", json={"email": "b@example.com", "password": "password123"})
    assert r.status_code == 200
    assert r.get_json()["token"]


def test_login_wrong_password(client):
    signup(client, "c@example.com", "password123", "student")
    r = client.post("/api/login", json={"email": "c@example.com", "password": "wrongpass"})
    assert r.status_code == 401


def test_login_unknown_email(client):
    r = client.post("/api/login", json={"email": "nope@example.com", "password": "password123"})
    assert r.status_code == 401


def test_protected_route_requires_bearer_token(client):
    r = client.get("/api/courses")
    assert r.status_code == 401


def test_protected_route_rejects_garbage_token(client):
    r = client.get("/api/courses", headers=auth_header("not-a-real-token"))
    assert r.status_code == 401


def test_jwt_sub_claim_is_a_string_on_the_wire(client):
    """Regression guard for the dev_collab_platform PyJWT `sub` bug
    (2026-07-11, PR #11): newer PyJWT rejects a non-string `sub`."""
    import jwt as pyjwt

    token, _ = signup(client, "d@example.com", "password123", "instructor")
    payload = pyjwt.decode(token, options={"verify_signature": False})
    assert isinstance(payload["sub"], str)
