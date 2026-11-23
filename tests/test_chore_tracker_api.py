"""Tests for chore_tracker_api."""
import pytest

from chore_tracker_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _chore(client, **overrides):
    payload = {
        "name": "Wash dishes",
        "assignees": ["Usman", "Ali", "Sara"],
        "frequency_days": 2,
    }
    payload.update(overrides)
    return client.post("/chores", json=payload)


def test_create_chore(client):
    resp = _chore(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["name"] == "Wash dishes"
    assert body["current_assignee"] == "Usman"
    assert body["frequency_days"] == 2
    assert body["last_done"] is None
    assert body["overdue"] is False
    assert body["times_done"] == 0


def test_create_chore_defaults_frequency(client):
    resp = client.post("/chores", json={"name": "Sweep", "assignees": ["Usman"]})
    assert resp.status_code == 201
    assert resp.get_json()["frequency_days"] == 7


def test_create_chore_validation(client):
    assert client.post("/chores", json={"name": "", "assignees": ["A"]}).status_code == 400
    assert client.post("/chores", json={"name": "X", "assignees": []}).status_code == 400
    assert client.post("/chores", json={"name": "X", "assignees": [""]}).status_code == 400
    assert client.post("/chores", json={"name": "X"}).status_code == 400
    assert _chore(client, frequency_days=0).status_code == 400
    assert _chore(client, frequency_days="daily").status_code == 400


def test_list_and_get_chore(client):
    _chore(client)
    _chore(client, name="Vacuum", assignees=["Ali"])
    assert len(client.get("/chores").get_json()) == 2
    assert client.get("/chores/1").get_json()["name"] == "Wash dishes"
    assert client.get("/chores/999").status_code == 404


def test_done_rotates_assignee(client):
    chore = _chore(client).get_json()
    first = client.post(f"/chores/{chore['id']}/done", json={}).get_json()
    assert first["current_assignee"] == "Ali"
    assert first["times_done"] == 1
    second = client.post(f"/chores/{chore['id']}/done", json={}).get_json()
    assert second["current_assignee"] == "Sara"
    third = client.post(f"/chores/{chore['id']}/done", json={}).get_json()
    assert third["current_assignee"] == "Usman"  # wraps around


def test_done_sets_due_date(client):
    chore = _chore(client).get_json()
    done = client.post(
        f"/chores/{chore['id']}/done", json={"date": "2026-11-23"}
    ).get_json()
    assert done["last_done"] == "2026-11-23"
    assert done["due_date"] == "2026-11-25"  # +2 days
    assert done["overdue"] is False


def test_done_validation(client):
    chore = _chore(client).get_json()
    assert client.post("/chores/999/done", json={}).status_code == 404
    assert client.post(f"/chores/{chore['id']}/done", json={"date": "tomorrow"}).status_code == 400


def test_overdue_when_past_due(client):
    chore = _chore(client, frequency_days=1).get_json()
    client.post(f"/chores/{chore['id']}/done", json={"date": "2026-11-10"})
    flagged = client.get(f"/chores/{chore['id']}").get_json()
    assert flagged["overdue"] is True
    assert flagged["due_date"] == "2026-11-11"


def test_not_overdue_before_due(client):
    chore = _chore(client, frequency_days=30).get_json()
    client.post(f"/chores/{chore['id']}/done", json={"date": "2026-11-23"})
    assert client.get(f"/chores/{chore['id']}").get_json()["overdue"] is False


def test_update_chore(client):
    chore = _chore(client).get_json()
    resp = client.put(
        f"/chores/{chore['id']}",
        json={"name": "Wash plates", "frequency_days": 3},
    )
    assert resp.status_code == 200
    assert resp.get_json()["name"] == "Wash plates"
    assert resp.get_json()["frequency_days"] == 3


def test_update_assignees_resets_turn(client):
    chore = _chore(client).get_json()
    client.post(f"/chores/{chore['id']}/done", json={})
    resp = client.put(
        f"/chores/{chore['id']}", json={"assignees": ["Zara", "Omar"]}
    )
    assert resp.get_json()["current_assignee"] == "Zara"


def test_update_validation(client):
    chore = _chore(client).get_json()
    assert client.put(f"/chores/{chore['id']}", json={"assignees": []}).status_code == 400
    assert client.put(f"/chores/{chore['id']}", json={"frequency_days": 0}).status_code == 400
    assert client.put("/chores/999", json={"name": "X"}).status_code == 404


def test_delete_chore(client):
    chore = _chore(client).get_json()
    assert client.delete(f"/chores/{chore['id']}").status_code == 200
    assert client.get(f"/chores/{chore['id']}").status_code == 404
    assert client.delete("/chores/999").status_code == 404
