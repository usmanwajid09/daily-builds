"""Tests for the reading_list_api Flask app, using Flask's test client."""
import pytest

from reading_list_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _add(client, title="Dune", author="Frank Herbert", total_pages=688, **extra):
    payload = {"title": title, "author": author, "total_pages": total_pages}
    payload.update(extra)
    return client.post("/books", json=payload)


def test_list_starts_empty(client):
    assert client.get("/books").get_json() == []


def test_create_book_returns_201_with_progress(client):
    resp = _add(client, pages_read=172)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "title": "Dune",
        "author": "Frank Herbert",
        "status": "to-read",
        "total_pages": 688,
        "pages_read": 172,
        "progress_pct": 25.0,
    }


def test_create_defaults_to_to_read_and_zero_pages(client):
    book = _add(client).get_json()
    assert book["status"] == "to-read"
    assert book["pages_read"] == 0
    assert book["progress_pct"] == 0.0


def test_create_missing_title_400(client):
    resp = client.post("/books", json={"author": "X", "total_pages": 10})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_missing_author_400(client):
    resp = client.post("/books", json={"title": "T", "total_pages": 10})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_zero_total_pages_400(client):
    assert _add(client, total_pages=0).status_code == 400


def test_create_pages_read_above_total_400(client):
    assert _add(client, pages_read=700).status_code == 400


def test_create_bad_status_400(client):
    assert _add(client, status="lost").status_code == 400


def test_create_finished_requires_full_pages_read(client):
    assert _add(client, status="finished", pages_read=10).status_code == 400
    resp = _add(client, status="finished", pages_read=688)
    assert resp.status_code == 201
    assert resp.get_json()["status"] == "finished"


def test_get_book_and_404(client):
    _add(client)
    assert client.get("/books/1").get_json()["title"] == "Dune"
    resp = client.get("/books/99")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_filter_by_author_case_insensitive(client):
    _add(client)
    _add(client, title="Foundation", author="Isaac Asimov", total_pages=296)
    found = client.get("/books", query_string={"author": "HERBERT"}).get_json()
    assert [b["title"] for b in found] == ["Dune"]
    assert client.get("/books", query_string={"author": "nobody"}).get_json() == []


def test_filter_by_status_and_bad_status_400(client):
    _add(client, status="reading", pages_read=10)
    _add(client, title="Foundation", author="Isaac Asimov", total_pages=296)
    assert len(client.get("/books", query_string={"status": "reading"}).get_json()) == 1
    assert client.get("/books", query_string={"status": "bogus"}).status_code == 400


def test_update_book(client):
    _add(client)
    resp = client.put("/books/1", json={"title": "Dune Messiah", "pages_read": 344})
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Dune Messiah"
    assert resp.get_json()["progress_pct"] == 50.0


def test_update_pages_read_above_total_400(client):
    _add(client)
    assert client.put("/books/1", json={"pages_read": 999}).status_code == 400


def test_update_unknown_book_404(client):
    assert client.put("/books/7", json={"title": "X"}).status_code == 404


def test_log_reading_moves_to_reading(client):
    _add(client)
    resp = client.post("/books/1/read", json={"pages": 50})
    assert resp.status_code == 200
    assert resp.get_json()["pages_read"] == 50
    assert resp.get_json()["status"] == "reading"


def test_log_reading_caps_and_auto_finishes(client):
    _add(client, total_pages=100, pages_read=90)
    resp = client.post("/books/1/read", json={"pages": 500})
    assert resp.get_json()["pages_read"] == 100
    assert resp.get_json()["status"] == "finished"
    assert resp.get_json()["progress_pct"] == 100.0


def test_log_reading_bad_pages_400(client):
    _add(client)
    assert client.post("/books/1/read", json={"pages": 0}).status_code == 400
    assert client.post("/books/1/read", json={"pages": -5}).status_code == 400
    assert client.post("/books/1/read", json={}).status_code == 400


def test_log_reading_unknown_book_404(client):
    assert client.post("/books/3/read", json={"pages": 5}).status_code == 404


def test_finish_book(client):
    _add(client)
    resp = client.post("/books/1/finish")
    assert resp.get_json()["status"] == "finished"
    assert resp.get_json()["pages_read"] == 688
    assert resp.get_json()["progress_pct"] == 100.0


def test_delete_book(client):
    _add(client)
    assert client.delete("/books/1").get_json() == {"deleted": 1}
    assert client.get("/books").get_json() == []
    assert client.delete("/books/1").status_code == 404


def test_stats(client):
    _add(client, total_pages=100, pages_read=50, status="reading")
    _add(client, title="Foundation", author="Isaac Asimov", total_pages=200)
    stats = client.get("/stats").get_json()
    assert stats["total_books"] == 2
    assert stats["by_status"] == {"to-read": 1, "reading": 1, "finished": 0}
    assert stats["total_pages"] == 300
    assert stats["pages_read"] == 50
    assert stats["overall_progress_pct"] == 16.7


def test_stats_empty(client):
    stats = client.get("/stats").get_json()
    assert stats == {
        "total_books": 0,
        "by_status": {"to-read": 0, "reading": 0, "finished": 0},
        "total_pages": 0,
        "pages_read": 0,
        "overall_progress_pct": 0.0,
    }
