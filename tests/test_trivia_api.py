"""Tests for the trivia_api Flask app."""

import pytest

from trivia_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _new_question(**overrides):
    payload = {
        "question": "What is 2 + 2?",
        "options": ["3", "4", "5"],
        "answer_index": 1,
        "category": "science",
        "difficulty": "easy",
    }
    payload.update(overrides)
    return payload


def test_seeded_questions_present(client):
    data = client.get("/questions").get_json()
    assert len(data) >= 12
    categories = {q["category"] for q in data}
    difficulties = {q["difficulty"] for q in data}
    assert len(categories) >= 4
    assert difficulties == {"easy", "medium", "hard"}


def test_list_hides_answer_index(client):
    data = client.get("/questions").get_json()
    assert data
    for question in data:
        assert "answer_index" not in question
        assert {"id", "question", "options", "category", "difficulty"} <= set(question)


def test_category_filter(client):
    data = client.get("/questions", query_string={"category": "history"}).get_json()
    assert data
    assert all(q["category"] == "history" for q in data)


def test_difficulty_filter(client):
    data = client.get("/questions", query_string={"difficulty": "hard"}).get_json()
    assert data
    assert all(q["difficulty"] == "hard" for q in data)


def test_invalid_difficulty_filter_400(client):
    response = client.get("/questions", query_string={"difficulty": "impossible"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_get_one_question(client):
    response = client.get("/questions/1")
    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == 1
    assert "answer_index" not in data
    assert data["question"]


def test_get_unknown_question_404(client):
    response = client.get("/questions/9999")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_add_question_201(client):
    response = client.post("/questions", json=_new_question())
    assert response.status_code == 201
    data = response.get_json()
    assert data["question"] == "What is 2 + 2?"
    assert data["answer_index"] == 1
    assert data["id"]
    # The new question is now listable.
    fetched = client.get(f"/questions/{data['id']}").get_json()
    assert fetched["question"] == "What is 2 + 2?"


def test_add_invalid_question_bad_answer_index_400(client):
    response = client.post("/questions", json=_new_question(answer_index=5))
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_add_invalid_question_too_few_options_400(client):
    response = client.post("/questions", json=_new_question(options=["only one"]))
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_add_invalid_question_bad_difficulty_400(client):
    response = client.post("/questions", json=_new_question(difficulty="expert"))
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_answer_correct_increments_score(client):
    created = client.post("/questions", json=_new_question()).get_json()
    response = client.post(
        "/answer",
        json={"player": "Ayesha", "question_id": created["id"], "answer_index": 1},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["correct"] is True
    assert data["correct_index"] == 1
    assert data["player"] == "Ayesha"
    assert data["score"] == 1
    assert data["answered"] == 1


def test_answer_wrong_keeps_score_but_increments_answered(client):
    created = client.post("/questions", json=_new_question()).get_json()
    client.post("/answer", json={"player": "Bilal", "question_id": created["id"], "answer_index": 1})
    response = client.post(
        "/answer",
        json={"player": "Bilal", "question_id": created["id"], "answer_index": 0},
    )
    data = response.get_json()
    assert data["correct"] is False
    assert data["score"] == 1
    assert data["answered"] == 2


def test_answer_unknown_question_404(client):
    response = client.post(
        "/answer",
        json={"player": "Ayesha", "question_id": 9999, "answer_index": 0},
    )
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_answer_index_out_of_range_400(client):
    response = client.post(
        "/answer",
        json={"player": "Ayesha", "question_id": 1, "answer_index": 99},
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_leaderboard_sorted_desc(client):
    created = client.post("/questions", json=_new_question()).get_json()
    qid = created["id"]
    # Zara: 2 correct. Ali: 1 correct out of 2. Bilal: 0 correct.
    for _ in range(2):
        client.post("/answer", json={"player": "Zara", "question_id": qid, "answer_index": 1})
    client.post("/answer", json={"player": "Ali", "question_id": qid, "answer_index": 1})
    client.post("/answer", json={"player": "Ali", "question_id": qid, "answer_index": 0})
    client.post("/answer", json={"player": "Bilal", "question_id": qid, "answer_index": 0})

    board = client.get("/scores").get_json()
    assert [row["player"] for row in board] == ["Zara", "Ali", "Bilal"]
    assert board[0]["score"] == 2 and board[0]["accuracy"] == 1.0
    assert board[1]["score"] == 1 and board[1]["accuracy"] == 0.5
    assert board[2]["score"] == 0 and board[2]["accuracy"] == 0.0


def test_player_score_404_for_unknown_player(client):
    response = client.get("/scores/Nobody")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_player_score_record(client):
    created = client.post("/questions", json=_new_question()).get_json()
    client.post("/answer", json={"player": "Usman", "question_id": created["id"], "answer_index": 1})
    client.post("/answer", json={"player": "Usman", "question_id": created["id"], "answer_index": 0})
    response = client.get("/scores/Usman")
    assert response.status_code == 200
    assert response.get_json() == {
        "player": "Usman",
        "score": 1,
        "answered": 2,
        "accuracy": 0.5,
    }
