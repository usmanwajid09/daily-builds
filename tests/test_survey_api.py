"""Tests for the survey_api Flask app, using Flask's test client."""
import pytest

from survey_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


QUESTIONS = [
    {"text": "Favourite topic?", "type": "choice",
     "options": ["Databases", "AI", "Networks"]},
    {"text": "Rate the course", "type": "rating"},
    {"text": "Any comments?", "type": "text"},
]


def _survey(client, title="Feedback", questions=None):
    return client.post(
        "/surveys",
        json={"title": title, "questions": questions if questions is not None
              else QUESTIONS},
    )


def _respond(client, survey_id, answers):
    return client.post(f"/surveys/{survey_id}/responses", json={"answers": answers})


GOOD = {"1": "AI", "2": 5, "3": "Great course"}


def test_list_starts_empty(client):
    assert client.get("/surveys").get_json() == []


def test_create_survey_returns_201(client):
    resp = _survey(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["title"] == "Feedback"
    assert body["response_count"] == 0
    assert body["questions"][0] == {
        "id": 1, "text": "Favourite topic?", "type": "choice",
        "options": ["Databases", "AI", "Networks"],
    }
    assert body["questions"][1] == {
        "id": 2, "text": "Rate the course", "type": "rating", "options": [],
    }


def test_create_validation_errors(client):
    assert _survey(client, "").status_code == 400
    assert _survey(client, "T", []).status_code == 400
    assert _survey(client, "T", "not a list").status_code == 400
    assert _survey(client, "T", [{"text": "Q", "type": "bogus"}]).status_code == 400
    assert _survey(client, "T", [{"text": "", "type": "text"}]).status_code == 400
    assert _survey(client, "T", [{"text": "Pick one", "type": "choice",
                                  "options": ["only one"]}]).status_code == 400
    assert _survey(client, "T", [{"text": "Pick one", "type": "choice"}]
                     ).status_code == 400


def test_get_survey_and_404(client):
    sid = _survey(client).get_json()["id"]
    assert client.get(f"/surveys/{sid}").get_json()["title"] == "Feedback"
    assert client.get("/surveys/999").status_code == 404


def test_submit_response_returns_201(client):
    sid = _survey(client).get_json()["id"]
    resp = _respond(client, sid, GOOD)
    assert resp.status_code == 201
    assert resp.get_json() == {"id": 1, "survey_id": sid, "answers": GOOD}
    assert client.get(f"/surveys/{sid}").get_json()["response_count"] == 1


def test_response_validation(client):
    sid = _survey(client).get_json()["id"]
    assert _respond(client, sid, {"1": "AI", "2": 5}).status_code == 400
    assert _respond(client, sid, {"1": "Wrong", "2": 5, "3": "x"}).status_code == 400
    assert _respond(client, sid, {"1": "AI", "2": 0, "3": "x"}).status_code == 400
    assert _respond(client, sid, {"1": "AI", "2": 6, "3": "x"}).status_code == 400
    assert _respond(client, sid, {"1": "AI", "2": 4.5, "3": "x"}).status_code == 400
    assert _respond(client, sid, {"1": "AI", "2": True, "3": "x"}).status_code == 400
    assert _respond(client, sid, {"1": "AI", "2": 5, "3": "  "}).status_code == 400
    assert _respond(client, sid, {"1": "AI", "2": 5, "3": "x", "9": "y"}
                     ).status_code == 400
    assert _respond(client, sid, "not an object").status_code == 400
    assert _respond(client, 999, GOOD).status_code == 404


def test_list_responses(client):
    sid = _survey(client).get_json()["id"]
    _respond(client, sid, GOOD)
    _respond(client, sid, {"1": "Networks", "2": 3, "3": "OK"})
    responses = client.get(f"/surveys/{sid}/responses").get_json()
    assert len(responses) == 2
    assert responses[0]["answers"]["1"] == "AI"
    assert client.get("/surveys/999/responses").status_code == 404


def test_summary_counts_and_average(client):
    sid = _survey(client).get_json()["id"]
    _respond(client, sid, GOOD)
    _respond(client, sid, {"1": "Databases", "2": 4, "3": "Faster pace"})
    _respond(client, sid, {"1": "AI", "2": 5, "3": "Loved it"})
    summary = client.get(f"/surveys/{sid}/summary").get_json()
    assert summary["response_count"] == 3
    q1, q2, q3 = summary["questions"]
    assert q1["counts"] == {"Databases": 1, "AI": 2, "Networks": 0}
    assert q2["average"] == 4.67
    assert q2["response_count"] == 3
    assert q3["answers"] == ["Great course", "Faster pace", "Loved it"]


def test_summary_with_no_responses(client):
    sid = _survey(client).get_json()["id"]
    summary = client.get(f"/surveys/{sid}/summary").get_json()
    assert summary["response_count"] == 0
    assert summary["questions"][0]["counts"] == {
        "Databases": 0, "AI": 0, "Networks": 0,
    }
    assert summary["questions"][1]["average"] is None
    assert summary["questions"][2]["answers"] == []
    assert client.get("/surveys/999/summary").status_code == 404


def test_delete_survey_removes_responses(client):
    sid = _survey(client).get_json()["id"]
    _respond(client, sid, GOOD)
    resp = client.delete(f"/surveys/{sid}")
    assert resp.get_json() == {"deleted": sid, "responses_deleted": 1}
    assert client.get(f"/surveys/{sid}").status_code == 404
    assert client.delete(f"/surveys/{sid}").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
