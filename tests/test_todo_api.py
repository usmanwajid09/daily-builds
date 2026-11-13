"""Tests for the todo_api Flask app, using Flask's test client."""
import pytest

from todo_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_list_starts_empty(client):
    assert client.get("/todos").get_json() == []


def test_create_todo_returns_201_with_id(client):
    resp = client.post("/todos", json={"title": "Buy milk"})
    assert resp.status_code == 201
    assert resp.get_json() == {"id": 1, "title": "Buy milk", "done": False}


def test_create_todo_trims_title_and_accepts_done(client):
    resp = client.post("/todos", json={"title": "  Read  ", "done": True})
    assert resp.get_json()["title"] == "Read"
    assert resp.get_json()["done"] is True


def test_create_requires_title(client):
    resp = client.post("/todos", json={})
    assert resp.status_code == 400
    assert "title" in resp.get_json()["error"]


def test_create_rejects_blank_title_and_non_bool_done(client):
    assert client.post("/todos", json={"title": "   "}).status_code == 400
    resp = client.post("/todos", json={"title": "x", "done": "yes"})
    assert resp.status_code == 400
    assert "boolean" in resp.get_json()["error"]


def test_create_rejects_non_json_body(client):
    resp = client.post("/todos", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_get_update_and_delete_round_trip(client):
    created = client.post("/todos", json={"title": "Write tests"}).get_json()
    todo_id = created["id"]
    assert client.get(f"/todos/{todo_id}").get_json()["title"] == "Write tests"
    updated = client.put(
        f"/todos/{todo_id}", json={"title": "Write more tests", "done": True}
    ).get_json()
    assert updated == {"id": todo_id, "title": "Write more tests", "done": True}
    assert client.delete(f"/todos/{todo_id}").get_json() == {"deleted": todo_id}
    assert client.get(f"/todos/{todo_id}").status_code == 404


def test_update_with_no_fields_is_400(client):
    todo_id = client.post("/todos", json={"title": "x"}).get_json()["id"]
    resp = client.put(f"/todos/{todo_id}", json={})
    assert resp.status_code == 400


def test_unknown_todo_is_404_json(client):
    for method in ("get", "put", "delete"):
        resp = getattr(client, method)("/todos/999", json={})
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]


def test_unknown_route_is_404_json(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
