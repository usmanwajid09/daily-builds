"""Tests for the contact_api Flask app, using Flask's test client."""
import pytest

from contact_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _submit(client, name="Ayesha", email="ayesha@example.com", message="Hello"):
    return client.post(
        "/messages", json={"name": name, "email": email, "message": message}
    )


def test_list_starts_empty(client):
    assert client.get("/messages").get_json() == []


def test_create_message_returns_201(client):
    resp = _submit(client)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "Ayesha",
        "email": "ayesha@example.com",
        "message": "Hello",
    }


def test_create_trims_whitespace(client):
    record = _submit(client, name="  Ayesha  ", email=" ayesha@example.com ").get_json()
    assert record["name"] == "Ayesha"
    assert record["email"] == "ayesha@example.com"


def test_create_rejects_empty_name(client):
    assert _submit(client, name="").status_code == 400
    assert _submit(client, name="   ").status_code == 400


def test_create_rejects_missing_name(client):
    resp = client.post(
        "/messages", json={"email": "a@b.com", "message": "Hi"}
    )
    assert resp.status_code == 400


def test_create_rejects_bad_email(client):
    for bad in ("not-an-email", "a@b", "a b@c.com", "@example.com", "a@"):
        assert _submit(client, email=bad).status_code == 400


def test_create_rejects_missing_email(client):
    resp = client.post("/messages", json={"name": "A", "message": "Hi"})
    assert resp.status_code == 400


def test_create_rejects_empty_message(client):
    assert _submit(client, message="").status_code == 400
    assert _submit(client, message="   ").status_code == 400


def test_create_rejects_too_long_message(client):
    assert _submit(client, message="x" * 5001).status_code == 400


def test_create_rejects_non_json_body(client):
    assert client.post("/messages", data="not json").status_code == 400


def test_get_message_by_id(client):
    created = _submit(client).get_json()
    fetched = client.get(f"/messages/{created['id']}").get_json()
    assert fetched == created


def test_get_unknown_message_404(client):
    resp = client.get("/messages/99")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_list_returns_all_in_order(client):
    _submit(client, name="First")
    _submit(client, name="Second")
    messages = client.get("/messages").get_json()
    assert [m["name"] for m in messages] == ["First", "Second"]


def test_delete_message(client):
    created = _submit(client).get_json()
    assert client.delete(f"/messages/{created['id']}").status_code == 200
    assert client.get("/messages").get_json() == []
    assert client.delete(f"/messages/{created['id']}").status_code == 404
