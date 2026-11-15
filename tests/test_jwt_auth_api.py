"""Tests for jwt_auth_api."""
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from jwt_auth_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _register(client, username="usman", password="hunter2secret"):
    return client.post(
        "/auth/register", json={"username": username, "password": password}
    )


def _login(client, username="usman", password="hunter2secret"):
    return client.post(
        "/auth/login", json={"username": username, "password": password}
    )


def _token(client):
    _register(client)
    return _login(client).get_json()["token"]


def test_register(client):
    resp = _register(client)
    assert resp.status_code == 201
    assert resp.get_json()["username"] == "usman"


def test_register_duplicate_username(client):
    _register(client)
    resp = _register(client)
    assert resp.status_code == 409
    assert "already taken" in resp.get_json()["error"]


def test_register_rejects_short_password(client):
    resp = _register(client, password="abc")
    assert resp.status_code == 400
    assert "password" in resp.get_json()["error"]


def test_register_rejects_missing_username(client):
    resp = client.post("/auth/register", json={"password": "hunter2secret"})
    assert resp.status_code == 400


def test_login_returns_token(client):
    _register(client)
    resp = _login(client)
    assert resp.status_code == 200
    token = resp.get_json()["token"]
    payload = jwt.decode(
        token, "dev-secret-change-me-in-production-0123456789abcdef", algorithms=["HS256"]
    )
    assert payload["sub"] == "usman"


def test_login_wrong_password(client):
    _register(client)
    resp = _login(client, password="wrongpassword")
    assert resp.status_code == 401
    assert "invalid username or password" in resp.get_json()["error"]


def test_login_unknown_user(client):
    resp = _login(client)
    assert resp.status_code == 401


def test_protected_with_valid_token(client):
    token = _token(client)
    resp = client.get(
        "/auth/protected", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert "usman" in resp.get_json()["message"]


def test_me_returns_username(client):
    token = _token(client)
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.get_json() == {"username": "usman"}


def test_protected_without_token(client):
    resp = client.get("/auth/protected")
    assert resp.status_code == 401
    assert "Authorization" in resp.get_json()["error"]


def test_protected_with_garbage_token(client):
    resp = client.get(
        "/auth/protected", headers={"Authorization": "Bearer not-a-token"}
    )
    assert resp.status_code == 401
    assert "invalid token" in resp.get_json()["error"]


def test_protected_with_expired_token(client):
    now = datetime.now(timezone.utc)
    expired = jwt.encode(
        {"sub": "usman", "iat": now - timedelta(hours=2),
         "exp": now - timedelta(hours=1)},
        "dev-secret-change-me-in-production-0123456789abcdef",
        algorithm="HS256",
    )
    resp = client.get(
        "/auth/protected", headers={"Authorization": f"Bearer {expired}"}
    )
    assert resp.status_code == 401
    assert "expired" in resp.get_json()["error"]


def test_protected_with_wrong_secret_token(client):
    token = jwt.encode(
        {"sub": "usman", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        "someone-elses-secret-that-is-long-enough-0123456789",
        algorithm="HS256",
    )
    resp = client.get(
        "/auth/protected", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 401


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
