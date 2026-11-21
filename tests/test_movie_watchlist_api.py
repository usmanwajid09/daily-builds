"""Tests for the movie_watchlist_api Flask app, using Flask's test client."""
import pytest

from movie_watchlist_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _movie(client, title="Dune: Part Two", genre="Sci-Fi", year=2024,
           **extra):
    payload = {"title": title, "genre": genre, "year": year}
    payload.update(extra)
    return client.post("/movies", json=payload)


def test_list_starts_empty(client):
    assert client.get("/movies").get_json() == []


def test_create_movie_returns_201(client):
    resp = _movie(client)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "title": "Dune: Part Two",
        "genre": "Sci-Fi",
        "year": 2024,
        "status": "to_watch",
        "rating": None,
    }


def test_create_watched_with_rating(client):
    body = _movie(client, title="Interstellar", year=2014,
                  status="watched", rating=5).get_json()
    assert body["status"] == "watched"
    assert body["rating"] == 5


def test_create_missing_title_400(client):
    resp = client.post("/movies", json={"genre": "Sci-Fi", "year": 2024})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_bad_year_400(client):
    assert _movie(client, year=1800).status_code == 400
    assert _movie(client, year=2031).status_code == 400
    assert _movie(client, year="2024").status_code == 400


def test_create_bad_status_400(client):
    assert _movie(client, status="loved").status_code == 400


def test_create_rating_on_unwatched_400(client):
    assert _movie(client, rating=4).status_code == 400
    assert _movie(client, status="watching", rating=4).status_code == 400


def test_create_rating_out_of_range_400(client):
    assert _movie(client, status="watched", rating=0).status_code == 400
    assert _movie(client, status="watched", rating=6).status_code == 400


def test_duplicate_title_409_case_insensitive(client):
    assert _movie(client).status_code == 201
    assert _movie(client, title="dune: part two").status_code == 409


def test_get_movie(client):
    _movie(client)
    resp = client.get("/movies/1")
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Dune: Part Two"


def test_get_missing_404(client):
    assert client.get("/movies/99").status_code == 404


def test_filter_by_genre(client):
    _movie(client, title="Dune", genre="Sci-Fi")
    _movie(client, title="Parasite", genre="Thriller", year=2019)
    rows = client.get("/movies?genre=sci-fi").get_json()
    assert [m["title"] for m in rows] == ["Dune"]


def test_filter_by_status(client):
    _movie(client, title="Dune")
    _movie(client, title="Interstellar", year=2014, status="watched",
           rating=5)
    rows = client.get("/movies?status=watched").get_json()
    assert [m["title"] for m in rows] == ["Interstellar"]


def test_filter_bad_status_400(client):
    assert client.get("/movies?status=loved").status_code == 400


def test_filter_by_min_rating(client):
    _movie(client, title="A", status="watched", rating=3)
    _movie(client, title="B", status="watched", rating=5)
    _movie(client, title="C")
    rows = client.get("/movies?min_rating=5").get_json()
    assert [m["title"] for m in rows] == ["B"]


def test_watch_marks_watched_with_rating(client):
    _movie(client)
    resp = client.post("/movies/1/watch", json={"rating": 4})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "watched"
    assert resp.get_json()["rating"] == 4


def test_watch_without_rating(client):
    _movie(client)
    assert client.post("/movies/1/watch", json={}).get_json()["status"] \
        == "watched"


def test_watch_already_watched_409(client):
    _movie(client, status="watched", rating=5)
    assert client.post("/movies/1/watch", json={}).status_code == 409


def test_watch_already_watched_can_update_rating(client):
    _movie(client, status="watched", rating=3)
    resp = client.post("/movies/1/watch", json={"rating": 5})
    assert resp.status_code == 200
    assert resp.get_json()["rating"] == 5


def test_watch_missing_404(client):
    assert client.post("/movies/99/watch", json={}).status_code == 404


def test_update_clears_rating_when_unwatching(client):
    _movie(client, status="watched", rating=5)
    resp = client.put("/movies/1", json={"status": "to_watch"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "to_watch"
    assert resp.get_json()["rating"] is None


def test_update_to_duplicate_title_409(client):
    _movie(client, title="Dune")
    _movie(client, title="Interstellar", year=2014)
    assert client.put("/movies/2", json={"title": "DUNE"}).status_code == 409


def test_update_nothing_400(client):
    _movie(client)
    assert client.put("/movies/1", json={}).status_code == 400


def test_update_missing_404(client):
    assert client.put("/movies/99", json={"title": "X"}).status_code == 404


def test_delete_movie(client):
    _movie(client)
    assert client.delete("/movies/1").get_json() == {"deleted": 1}
    assert client.get("/movies/1").status_code == 404


def test_delete_missing_404(client):
    assert client.delete("/movies/99").status_code == 404


def test_stats_empty(client):
    stats = client.get("/stats").get_json()
    assert stats["total"] == 0
    assert stats["average_rating"] is None
    assert stats["by_status"] == {"to_watch": 0, "watching": 0,
                                  "watched": 0}


def test_stats_counts_and_average(client):
    _movie(client, title="A", genre="Sci-Fi", status="watched", rating=4)
    _movie(client, title="B", genre="Sci-Fi", status="watched", rating=5)
    _movie(client, title="C", genre="Action", year=2008,
           status="watching")
    stats = client.get("/stats").get_json()
    assert stats["total"] == 3
    assert stats["by_status"] == {"to_watch": 0, "watching": 1,
                                  "watched": 2}
    assert stats["average_rating"] == 4.5
    assert stats["rated_films"] == 2
    assert stats["by_genre"] == {"Sci-Fi": 2, "Action": 1}


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
