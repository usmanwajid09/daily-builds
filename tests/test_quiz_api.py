"""Tests for quiz_api."""
import pytest

from quiz_api.app import create_app

QUIZ = {
    "title": "Python basics",
    "questions": [
        {"question": "What does len([1, 2, 3]) return?",
         "options": ["2", "3", "4"], "correct_index": 1},
        {"question": "Which keyword defines a function?",
         "options": ["def", "func"], "correct_index": 0},
    ],
}


@pytest.fixture
def client():
    return create_app().test_client()


def _create(client, payload=None):
    return client.post("/quizzes", json=payload if payload is not None else QUIZ)


def test_create_quiz(client):
    resp = _create(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["question_count"] == 2
    assert data["questions"][0]["options"] == ["2", "3", "4"]


def test_create_hides_correct_answers(client):
    data = _create(client).get_json()
    assert "correct_index" not in data["questions"][0]


def test_list_quizzes(client):
    _create(client)
    _create(client, {**QUIZ, "title": "Second quiz"})
    listed = client.get("/quizzes").get_json()
    assert len(listed) == 2
    assert [q["id"] for q in listed] == [1, 2]


def test_get_quiz(client):
    _create(client)
    data = client.get("/quizzes/1").get_json()
    assert data["title"] == "Python basics"


def test_submit_perfect_score(client):
    _create(client)
    result = client.post("/quizzes/1/submit", json={"answers": [1, 0]}).get_json()
    assert result["score"] == 2
    assert result["total"] == 2
    assert result["percentage"] == 100.0
    assert all(r["correct"] for r in result["results"])


def test_submit_partial_score(client):
    _create(client)
    result = client.post("/quizzes/1/submit", json={"answers": [0, 0]}).get_json()
    assert result["score"] == 1
    assert result["percentage"] == 50.0
    assert result["results"][0]["correct"] is False


def test_submit_wrong_answer_count_400(client):
    _create(client)
    resp = client.post("/quizzes/1/submit", json={"answers": [1]})
    assert resp.status_code == 400


def test_submit_out_of_range_answer_400(client):
    _create(client)
    resp = client.post("/quizzes/1/submit", json={"answers": [5, 0]})
    assert resp.status_code == 400


def test_create_missing_title_400(client):
    resp = _create(client, {"questions": QUIZ["questions"]})
    assert resp.status_code == 400


def test_create_empty_questions_400(client):
    resp = _create(client, {"title": "Empty", "questions": []})
    assert resp.status_code == 400


def test_create_one_option_400(client):
    bad = {"title": "Bad", "questions": [
        {"question": "Q?", "options": ["only"], "correct_index": 0}]}
    assert _create(client, bad).status_code == 400


def test_create_bad_correct_index_400(client):
    bad = {"title": "Bad", "questions": [
        {"question": "Q?", "options": ["a", "b"], "correct_index": 7}]}
    assert _create(client, bad).status_code == 400


def test_unknown_quiz_404(client):
    assert client.get("/quizzes/99").status_code == 404
    assert client.post("/quizzes/99/submit", json={"answers": []}).status_code == 404


def test_delete_quiz(client):
    _create(client)
    assert client.delete("/quizzes/1").status_code == 200
    assert client.get("/quizzes/1").status_code == 404
