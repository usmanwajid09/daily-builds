"""Tests for the ticket_api Flask app, using Flask's test client."""
import pytest

from ticket_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _ticket(client, title="Bug", **extra):
    payload = {"title": title}
    payload.update(extra)
    return client.post("/tickets", json=payload)


def _move(client, ticket_id, status):
    return client.post(f"/tickets/{ticket_id}/status", json={"status": status})


def test_list_starts_empty(client):
    assert client.get("/tickets").get_json() == []


def test_create_ticket_returns_201(client):
    resp = _ticket(client, "Login crash", description="500 on submit",
                   priority="high", assignee="ali")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1, "title": "Login crash", "description": "500 on submit",
        "priority": "high", "assignee": "ali", "status": "open",
    }


def test_create_defaults(client):
    ticket = _ticket(client, "Bare").get_json()
    assert ticket["status"] == "open"
    assert ticket["priority"] == "medium"
    assert ticket["description"] == ""
    assert ticket["assignee"] is None


def test_create_validation_errors(client):
    assert _ticket(client, "").status_code == 400
    assert _ticket(client, "   ").status_code == 400
    assert _ticket(client, 42).status_code == 400
    assert _ticket(client, "T", priority="critical").status_code == 400
    assert _ticket(client, "T", description=7).status_code == 400
    assert _ticket(client, "T", assignee="").status_code == 400
    assert _ticket(client, "T", assignee=5).status_code == 400


def test_get_ticket_and_404(client):
    tid = _ticket(client, "Hello").get_json()["id"]
    assert client.get(f"/tickets/{tid}").get_json()["title"] == "Hello"
    assert client.get("/tickets/999").status_code == 404


def test_full_workflow_open_to_closed(client):
    tid = _ticket(client, "Flow").get_json()["id"]
    for step in ("in_progress", "resolved", "closed"):
        resp = _move(client, tid, step)
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json()["status"] == step


def test_direct_close_from_open(client):
    tid = _ticket(client, "Spam").get_json()["id"]
    assert _move(client, tid, "closed").get_json()["status"] == "closed"


def test_illegal_transitions_rejected(client):
    tid = _ticket(client, "Jump").get_json()["id"]
    assert _move(client, tid, "resolved").status_code == 400
    assert _move(client, tid, "closed").status_code == 200
    assert _move(client, tid, "in_progress").status_code == 400
    assert _move(client, tid, "open").status_code == 400
    assert _move(client, tid, "bogus").status_code == 400


def test_reopen_paths(client):
    tid = _ticket(client, "Reopen").get_json()["id"]
    _move(client, tid, "in_progress")
    assert _move(client, tid, "open").get_json()["status"] == "open"
    _move(client, tid, "in_progress")
    _move(client, tid, "resolved")
    assert _move(client, tid, "in_progress").get_json()["status"] == "in_progress"


def test_same_status_is_rejected(client):
    tid = _ticket(client, "Same").get_json()["id"]
    assert _move(client, tid, "open").status_code == 400
    assert _move(client, 999, "closed").status_code == 404


def test_update_ticket(client):
    tid = _ticket(client, "Old", priority="low").get_json()["id"]
    updated = client.put(
        f"/tickets/{tid}",
        json={"title": "New", "priority": "urgent", "assignee": "usman"},
    ).get_json()
    assert updated["title"] == "New"
    assert updated["priority"] == "urgent"
    assert updated["assignee"] == "usman"
    assert updated["status"] == "open"
    unassigned = client.put(f"/tickets/{tid}", json={"assignee": None}).get_json()
    assert unassigned["assignee"] is None


def test_update_validation(client):
    tid = _ticket(client, "T").get_json()["id"]
    assert client.put(f"/tickets/{tid}", json={}).status_code == 400
    assert client.put(f"/tickets/{tid}", json={"title": " "}).status_code == 400
    assert client.put(f"/tickets/{tid}", json={"priority": "x"}).status_code == 400
    assert client.put(
        f"/tickets/{tid}", json={"status": "closed"}
    ).status_code == 400
    assert client.put("/tickets/999", json={"title": "x"}).status_code == 404


def test_filters(client):
    _ticket(client, "A", priority="high", assignee="ali")
    tid_b = _ticket(client, "B", priority="low", assignee="usman").get_json()["id"]
    _move(client, tid_b, "closed")
    assert [t["title"] for t in client.get(
        "/tickets", query_string={"status": "closed"}).get_json()] == ["B"]
    assert [t["title"] for t in client.get(
        "/tickets", query_string={"priority": "high"}).get_json()] == ["A"]
    assert [t["title"] for t in client.get(
        "/tickets", query_string={"assignee": "usman"}).get_json()] == ["B"]
    assert client.get("/tickets", query_string={"status": "nope"}).status_code == 400
    assert client.get(
        "/tickets", query_string={"priority": "nope"}).status_code == 400


def test_delete_ticket(client):
    tid = _ticket(client, "Gone").get_json()["id"]
    assert client.delete(f"/tickets/{tid}").get_json() == {"deleted": tid}
    assert client.get(f"/tickets/{tid}").status_code == 404
    assert client.delete(f"/tickets/{tid}").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
