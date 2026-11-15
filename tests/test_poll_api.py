"""Tests for poll_api."""
import pytest

from poll_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _create_poll(client):
    return client.post(
        "/polls",
        json={
            "question": "Best language for a quick API?",
            "options": ["Python", "Go", "JavaScript"],
        },
    )


def test_create_poll(client):
    resp = _create_poll(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["question"] == "Best language for a quick API?"
    assert data["votes"] == {"Python": 0, "Go": 0, "JavaScript": 0}
    assert data["total_votes"] == 0


def test_create_rejects_too_few_options(client):
    resp = client.post(
        "/polls", json={"question": "Only one?", "options": ["Python"]}
    )
    assert resp.status_code == 400


def test_create_rejects_duplicate_options(client):
    resp = client.post(
        "/polls", json={"question": "Dup?", "options": ["Python", "Python"]}
    )
    assert resp.status_code == 400
    assert "unique" in resp.get_json()["error"]


def test_create_rejects_missing_question(client):
    resp = client.post("/polls", json={"options": ["A", "B"]})
    assert resp.status_code == 400


def test_vote_and_results(client):
    poll_id = _create_poll(client).get_json()["id"]
    resp = client.post(
        f"/polls/{poll_id}/vote", json={"voter_id": "alice", "option": "Python"}
    )
    assert resp.status_code == 201
    client.post(f"/polls/{poll_id}/vote", json={"voter_id": "bob", "option": "Go"})
    client.post(
        f"/polls/{poll_id}/vote", json={"voter_id": "carol", "option": "Python"}
    )
    results = client.get(f"/polls/{poll_id}/results").get_json()
    assert results["total_votes"] == 3
    by_option = {r["option"]: r for r in results["results"]}
    assert by_option["Python"] == {"option": "Python", "votes": 2, "percentage": 66.7}
    assert by_option["Go"]["votes"] == 1
    assert by_option["Go"]["percentage"] == 33.3
    assert by_option["JavaScript"]["votes"] == 0
    assert by_option["JavaScript"]["percentage"] == 0.0


def test_vote_twice_rejected(client):
    poll_id = _create_poll(client).get_json()["id"]
    client.post(f"/polls/{poll_id}/vote", json={"voter_id": "alice", "option": "Python"})
    resp = client.post(
        f"/polls/{poll_id}/vote", json={"voter_id": "alice", "option": "Go"}
    )
    assert resp.status_code == 409
    assert "already voted" in resp.get_json()["error"]


def test_vote_invalid_option(client):
    poll_id = _create_poll(client).get_json()["id"]
    resp = client.post(
        f"/polls/{poll_id}/vote", json={"voter_id": "alice", "option": "Rust"}
    )
    assert resp.status_code == 400


def test_vote_missing_voter_id(client):
    poll_id = _create_poll(client).get_json()["id"]
    resp = client.post(f"/polls/{poll_id}/vote", json={"option": "Python"})
    assert resp.status_code == 400


def test_vote_missing_poll_404(client):
    resp = client.post(
        "/polls/99/vote", json={"voter_id": "alice", "option": "Python"}
    )
    assert resp.status_code == 404


def test_results_empty_poll(client):
    poll_id = _create_poll(client).get_json()["id"]
    results = client.get(f"/polls/{poll_id}/results").get_json()
    assert results["total_votes"] == 0
    assert all(r["percentage"] == 0.0 for r in results["results"])


def test_get_poll(client):
    poll_id = _create_poll(client).get_json()["id"]
    fetched = client.get(f"/polls/{poll_id}").get_json()
    assert fetched["question"] == "Best language for a quick API?"
    assert fetched["total_votes"] == 0


def test_get_missing_poll_404(client):
    assert client.get("/polls/99").status_code == 404


def test_delete_poll(client):
    poll_id = _create_poll(client).get_json()["id"]
    resp = client.delete(f"/polls/{poll_id}")
    assert resp.status_code == 200
    assert client.get("/polls").get_json() == []
    assert client.delete(f"/polls/{poll_id}").status_code == 404
