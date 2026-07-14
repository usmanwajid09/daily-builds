import pytest

from fintech_app import create_app


@pytest.fixture
def app():
    return create_app(db_path=":memory:", jwt_secret="test-secret")


@pytest.fixture
def client(app):
    return app.test_client()


def signup(client, email="user@example.com", password="password123"):
    r = client.post("/api/signup", json={"email": email, "password": password})
    assert r.status_code == 201, r.get_json()
    return r.get_json()["token"], r.get_json()["user"]["id"]


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}
