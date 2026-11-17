"""Tests for leaderboard_api."""
import pytest

from leaderboard_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _score(client, player="Ayesha", game="snake", score=100):
    return client.post("/scores", json={
        "player": player, "game": game, "score": score})


def _seed(client):
    _score(client, "Ayesha", "snake", 1500)
    _score(client, "Bilal", "snake", 2200)
    _score(client, "Ayesha", "snake", 2200)
    _score(client, "Chen", "snake", 900)
    _score(client, "Bilal", "tetris", 3100)


def test_submit_score(client):
    resp = _score(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["player"] == "Ayesha"
    assert data["score"] == 100
    assert data["submitted_at"] == 1


def test_ranking_highest_first(client):
    _seed(client)
    board = client.get("/leaderboard/snake").get_json()
    assert [r["score"] for r in board] == [2200, 2200, 1500, 900]
    assert [r["rank"] for r in board] == [1, 2, 3, 4]


def test_tie_broken_by_earliest_submission(client):
    _seed(client)
    board = client.get("/leaderboard/snake").get_json()
    assert board[0]["player"] == "Bilal"   # reached 2200 first
    assert board[1]["player"] == "Ayesha"  # reached 2200 second


def test_leaderboard_all_games(client):
    _seed(client)
    board = client.get("/leaderboard").get_json()
    assert board[0]["game"] == "tetris"
    assert board[0]["score"] == 3100
    assert len(board) == 5


def test_leaderboard_game_filter(client):
    _seed(client)
    board = client.get("/leaderboard?game=tetris").get_json()
    assert len(board) == 1
    assert board[0]["player"] == "Bilal"


def test_top_n(client):
    _seed(client)
    board = client.get("/leaderboard/snake?top=2").get_json()
    assert len(board) == 2
    assert board[0]["score"] == 2200
    assert board[1]["score"] == 2200


def test_player_best_per_game(client):
    _seed(client)
    best = client.get("/players/Ayesha/best").get_json()
    assert best["player"] == "Ayesha"
    assert best["best"] == [{"game": "snake", "score": 2200}]


def test_player_best_single_game(client):
    _seed(client)
    best = client.get("/players/Bilal/best?game=tetris").get_json()
    assert best["best"] == [{"game": "tetris", "score": 3100}]


def test_games_list(client):
    _seed(client)
    assert client.get("/games").get_json() == ["snake", "tetris"]


def test_submit_missing_score_400(client):
    resp = client.post("/scores", json={"player": "Dan", "game": "snake"})
    assert resp.status_code == 400


def test_submit_non_numeric_score_400(client):
    resp = client.post("/scores", json={
        "player": "Dan", "game": "snake", "score": "high"})
    assert resp.status_code == 400


def test_submit_missing_player_400(client):
    resp = client.post("/scores", json={"game": "snake", "score": 10})
    assert resp.status_code == 400


def test_unknown_game_404(client):
    _seed(client)
    resp = client.get("/leaderboard/chess")
    assert resp.status_code == 404


def test_unknown_player_404(client):
    _seed(client)
    resp = client.get("/players/Nobody/best")
    assert resp.status_code == 404


def test_bad_top_400(client):
    _seed(client)
    assert client.get("/leaderboard?top=0").status_code == 400
    assert client.get("/leaderboard?top=abc").status_code == 400
