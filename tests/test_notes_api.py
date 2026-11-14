"""Tests for the notes_api Flask app, using Flask's test client."""
import pytest

from notes_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _create(client, title="Note", body="", tags=None):
    payload = {"title": title, "body": body}
    if tags is not None:
        payload["tags"] = tags
    return client.post("/notes", json=payload)


def test_list_starts_empty(client):
    assert client.get("/notes").get_json() == []


def test_create_note_returns_201(client):
    resp = _create(client, "Grocery", "Milk, eggs", ["home"])
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1, "title": "Grocery", "body": "Milk, eggs", "tags": ["home"],
    }


def test_create_defaults_body_and_tags(client):
    note = _create(client, "Bare").get_json()
    assert note["body"] == "" and note["tags"] == []


def test_create_requires_non_empty_title(client):
    assert _create(client, "").status_code == 400
    assert _create(client, "   ").status_code == 400
    assert client.post("/notes", json={"body": "x"}).status_code == 400
    assert client.post("/notes", json={"title": 7}).status_code == 400


def test_create_rejects_bad_body_and_tags(client):
    assert _create(client, "T", body=5).status_code == 400
    assert _create(client, "T", tags="home").status_code == 400
    assert _create(client, "T", tags=["ok", ""]).status_code == 400
    assert _create(client, "T", tags=[1]).status_code == 400


def test_create_dedupes_and_trims_tags(client):
    note = _create(client, "T", tags=[" home ", "home", "code"]).get_json()
    assert note["tags"] == ["home", "code"]


def test_get_note_and_404(client):
    note_id = _create(client, "Hello").get_json()["id"]
    assert client.get(f"/notes/{note_id}").get_json()["title"] == "Hello"
    assert client.get("/notes/999").status_code == 404


def test_search_matches_title_or_body_case_insensitively(client):
    _create(client, "Flask tips", "Use create_app")
    _create(client, "Grocery", "Buy MILK")
    _create(client, "Other", "nothing here")
    assert [n["title"] for n in client.get("/notes", query_string={"search": "milk"}).get_json()] == ["Grocery"]
    assert [n["title"] for n in client.get("/notes", query_string={"search": "FLASK"}).get_json()] == ["Flask tips"]
    assert client.get("/notes", query_string={"search": "zzz"}).get_json() == []


def test_tag_filter(client):
    _create(client, "A", tags=["code", "flask"])
    _create(client, "B", tags=["home"])
    assert [n["title"] for n in client.get("/notes", query_string={"tag": "code"}).get_json()] == ["A"]
    assert client.get("/notes", query_string={"tag": "missing"}).get_json() == []


def test_search_and_tag_combine(client):
    _create(client, "Flask guide", "milk", ["code"])
    _create(client, "Flask other", "nope", ["home"])
    found = client.get(
        "/notes", query_string={"search": "flask", "tag": "code"}
    ).get_json()
    assert [n["title"] for n in found] == ["Flask guide"]


def test_update_round_trip(client):
    note_id = _create(client, "Old", "old body", ["a"]).get_json()["id"]
    updated = client.put(
        f"/notes/{note_id}",
        json={"title": "New", "body": "new body", "tags": ["b"]},
    ).get_json()
    assert updated == {"id": note_id, "title": "New", "body": "new body", "tags": ["b"]}
    assert client.put(f"/notes/{note_id}", json={}).status_code == 400
    assert client.put(f"/notes/{note_id}", json={"title": "  "}).status_code == 400
    assert client.put("/notes/999", json={"title": "x"}).status_code == 404


def test_delete_note(client):
    note_id = _create(client, "Gone").get_json()["id"]
    assert client.delete(f"/notes/{note_id}").get_json() == {"deleted": note_id}
    assert client.get(f"/notes/{note_id}").status_code == 404
    assert client.delete("/notes/999").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
