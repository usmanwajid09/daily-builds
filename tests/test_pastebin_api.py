"""Tests for the pastebin_api Flask app, using Flask's test client."""
import time

import pytest

from pastebin_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _create(client, content="hello", expires_in=None):
    payload = {"content": content}
    if expires_in is not None:
        payload["expires_in"] = expires_in
    return client.post("/snippets", json=payload)


def test_list_starts_empty(client):
    assert client.get("/snippets").get_json() == []


def test_create_snippet_returns_201(client):
    resp = _create(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["content"] == "hello"
    assert body["expires_at"] is None
    assert body["created_at"] > 0


def test_create_with_expiry_sets_expires_at(client):
    before = time.time()
    body = _create(client, expires_in=3600).get_json()
    assert before < body["expires_at"] <= before + 3600 + 1


def test_create_rejects_empty_content(client):
    assert _create(client, content="").status_code == 400
    assert _create(client, content="   ").status_code == 400


def test_create_rejects_missing_content(client):
    assert client.post("/snippets", json={}).status_code == 400


def test_create_rejects_non_json_body(client):
    assert client.post("/snippets", data="not json").status_code == 400


def test_create_rejects_bad_expiry(client):
    assert _create(client, expires_in=0).status_code == 400
    assert _create(client, expires_in=-10).status_code == 400
    assert _create(client, expires_in="soon").status_code == 400
    assert _create(client, expires_in=1.5).status_code == 400


def test_get_snippet_by_id(client):
    created = _create(client).get_json()
    fetched = client.get(f"/snippets/{created['id']}").get_json()
    assert fetched == created


def test_get_unknown_snippet_404(client):
    resp = client.get("/snippets/99")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_expired_snippet_returns_410(client):
    created = _create(client, expires_in=1).get_json()
    time.sleep(1.2)
    resp = client.get(f"/snippets/{created['id']}")
    assert resp.status_code == 410
    assert "expired" in resp.get_json()["error"]


def test_expired_snippet_hidden_from_list(client):
    _create(client, content="forever")
    short = _create(client, content="soon gone", expires_in=1).get_json()
    assert len(client.get("/snippets").get_json()) == 2
    time.sleep(1.2)
    live = client.get("/snippets").get_json()
    assert len(live) == 1
    assert live[0]["content"] == "forever"


def test_delete_snippet(client):
    created = _create(client).get_json()
    assert client.delete(f"/snippets/{created['id']}").status_code == 200
    assert client.get("/snippets").get_json() == []
    assert client.delete(f"/snippets/{created['id']}").status_code == 404


def test_list_returns_all_live_in_order(client):
    _create(client, content="one")
    _create(client, content="two")
    snippets = client.get("/snippets").get_json()
    assert [s["content"] for s in snippets] == ["one", "two"]
