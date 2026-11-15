"""Tests for quote_api."""
import pytest

from quote_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _add(client, text="Talk is cheap. Show me the code.",
        author="Linus Torvalds", tags=None):
    payload = {"text": text, "author": author}
    if tags is not None:
        payload["tags"] = tags
    return client.post("/quotes", json=payload)


def test_add_quote(client):
    resp = _add(client, tags=["code"])
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["author"] == "Linus Torvalds"
    assert data["tags"] == ["code"]


def test_add_defaults_tags_to_empty(client):
    assert _add(client).get_json()["tags"] == []


def test_list_quotes(client):
    _add(client)
    _add(client, text="Simplicity is the soul of efficiency.",
         author="Austin Freeman")
    listed = client.get("/quotes").get_json()
    assert len(listed) == 2
    assert [q["id"] for q in listed] == [1, 2]


def test_filter_by_author(client):
    _add(client)
    _add(client, text="Simplicity is the soul of efficiency.",
         author="Austin Freeman")
    found = client.get("/quotes?author=torvalds").get_json()
    assert len(found) == 1
    assert found[0]["author"] == "Linus Torvalds"


def test_filter_by_tag(client):
    _add(client, tags=["code", "programming"])
    _add(client, text="Simplicity is the soul of efficiency.",
         author="Austin Freeman", tags=["simplicity"])
    found = client.get("/quotes?tag=code").get_json()
    assert len(found) == 1
    assert found[0]["author"] == "Linus Torvalds"
    assert client.get("/quotes?tag=nope").get_json() == []


def test_random_quote(client):
    _add(client)
    _add(client, text="Simplicity is the soul of efficiency.",
         author="Austin Freeman")
    random_quote = client.get("/quotes/random").get_json()
    assert random_quote["author"] in ("Linus Torvalds", "Austin Freeman")


def test_random_quote_empty_store_404(client):
    resp = client.get("/quotes/random")
    assert resp.status_code == 404
    assert "no quotes" in resp.get_json()["error"]


def test_get_one_quote(client):
    created = _add(client).get_json()
    fetched = client.get(f"/quotes/{created['id']}").get_json()
    assert fetched["text"].startswith("Talk is cheap")


def test_get_missing_quote_404(client):
    resp = client.get("/quotes/99")
    assert resp.status_code == 404


def test_add_rejects_missing_text(client):
    resp = client.post("/quotes", json={"author": "Someone"})
    assert resp.status_code == 400
    assert "text" in resp.get_json()["error"]


def test_add_rejects_missing_author(client):
    resp = client.post("/quotes", json={"text": "Something wise"})
    assert resp.status_code == 400


def test_add_rejects_bad_tags(client):
    resp = _add(client, tags="code")
    assert resp.status_code == 400


def test_delete_quote(client):
    created = _add(client).get_json()
    resp = client.delete(f"/quotes/{created['id']}")
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] == created["id"]
    assert client.get("/quotes").get_json() == []


def test_delete_missing_quote_404(client):
    assert client.delete("/quotes/99").status_code == 404
