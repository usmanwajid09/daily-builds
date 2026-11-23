"""Tests for goal_tracker_api."""
import pytest

from goal_tracker_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _goal(client, **overrides):
    payload = {"title": "Read 12 books", "target": 12, "deadline": "2026-12-31"}
    payload.update(overrides)
    return client.post("/goals", json=payload)


def test_create_goal(client):
    resp = _goal(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["title"] == "Read 12 books"
    assert body["target"] == 12
    assert body["deadline"] == "2026-12-31"
    assert body["progress"] == 0
    assert body["milestones"] == []


def test_create_goal_missing_title(client):
    assert _goal(client, title="").status_code == 400
    assert client.post("/goals", json={"target": 5, "deadline": "2026-12-31"}).status_code == 400


def test_create_goal_bad_target(client):
    assert _goal(client, target=0).status_code == 400
    assert _goal(client, target=-3).status_code == 400
    assert _goal(client, target="many").status_code == 400


def test_create_goal_bad_deadline(client):
    assert _goal(client, deadline="31-12-2026").status_code == 400
    assert _goal(client, deadline="2026-02-30").status_code == 400
    assert _goal(client, deadline=None).status_code == 400


def test_create_goal_non_json_body(client):
    resp = client.post("/goals", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_list_goals(client):
    _goal(client)
    _goal(client, title="Run 100 km", target=100)
    listed = client.get("/goals").get_json()
    assert len(listed) == 2
    assert [g["title"] for g in listed] == ["Read 12 books", "Run 100 km"]


def test_get_goal_and_404(client):
    goal = _goal(client).get_json()
    assert client.get(f"/goals/{goal['id']}").status_code == 200
    assert client.get("/goals/999").status_code == 404


def test_update_progress(client):
    goal = _goal(client).get_json()
    resp = client.put(f"/goals/{goal['id']}/progress", json={"progress": 40})
    assert resp.status_code == 200
    assert resp.get_json()["progress"] == 40


def test_update_progress_out_of_range(client):
    goal = _goal(client).get_json()
    assert client.put(f"/goals/{goal['id']}/progress", json={"progress": 101}).status_code == 400
    assert client.put(f"/goals/{goal['id']}/progress", json={"progress": -1}).status_code == 400
    assert client.put(f"/goals/{goal['id']}/progress", json={}).status_code == 400


def test_update_progress_missing_goal(client):
    assert client.put("/goals/999/progress", json={"progress": 10}).status_code == 404


def test_milestone_progress_math(client):
    goal = _goal(client).get_json()
    gid = goal["id"]
    client.post(f"/goals/{gid}/milestones", json={"title": "Book 1"})
    client.post(f"/goals/{gid}/milestones", json={"title": "Book 2"})
    client.post(f"/goals/{gid}/milestones", json={"title": "Book 3"})
    client.post(f"/goals/{gid}/milestones", json={"title": "Book 4"})
    client.put(f"/goals/{gid}/milestones/1", json={"done": True})
    client.put(f"/goals/{gid}/milestones/2", json={"done": True})
    updated = client.get(f"/goals/{gid}").get_json()
    assert updated["progress"] == 50.0
    assert len(updated["milestones"]) == 4


def test_milestone_overrides_stored_progress(client):
    goal = _goal(client).get_json()
    gid = goal["id"]
    client.put(f"/goals/{gid}/progress", json={"progress": 90})
    client.post(f"/goals/{gid}/milestones", json={"title": "Only step"})
    assert client.get(f"/goals/{gid}").get_json()["progress"] == 0


def test_add_milestone_validation(client):
    goal = _goal(client).get_json()
    assert client.post(f"/goals/{goal['id']}/milestones", json={"title": ""}).status_code == 400
    assert client.post("/goals/999/milestones", json={"title": "x"}).status_code == 404


def test_update_milestone_404_and_bad_done(client):
    goal = _goal(client).get_json()
    gid = goal["id"]
    client.post(f"/goals/{gid}/milestones", json={"title": "Step"})
    assert client.put(f"/goals/{gid}/milestones/999", json={"done": True}).status_code == 404
    assert client.put(f"/goals/{gid}/milestones/1", json={"done": "yes"}).status_code == 400


def test_delete_goal(client):
    goal = _goal(client).get_json()
    assert client.delete(f"/goals/{goal['id']}").status_code == 200
    assert client.get(f"/goals/{goal['id']}").status_code == 404
    assert client.delete("/goals/999").status_code == 404
