"""Tests for the forum_api Flask app, using Flask's test client."""
import pytest

from forum_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _thread(client, title="Best Python tips", author="Usman"):
    return client.post("/threads", json={"title": title, "author": author})


def _post(client, thread_id, author="Usman", content="Hello"):
    return client.post(
        f"/threads/{thread_id}/posts", json={"author": author, "content": content}
    )


def test_list_starts_empty(client):
    assert client.get("/threads").get_json() == []


def test_create_thread_returns_201(client):
    resp = _thread(client, "Best Python tips", "Usman")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "title": "Best Python tips",
        "author": "Usman",
        "status": "open",
        "post_count": 0,
    }


def test_create_thread_requires_title_and_author(client):
    assert client.post("/threads", json={"title": "", "author": "Usman"}).status_code == 400
    assert client.post("/threads", json={"title": "T", "author": "  "}).status_code == 400
    assert client.post("/threads", json={"title": "T"}).status_code == 400
    assert client.post("/threads", json={}).status_code == 400


def test_get_thread_and_404(client):
    thread_id = _thread(client).get_json()["id"]
    assert client.get(f"/threads/{thread_id}").get_json()["title"] == "Best Python tips"
    assert client.get("/threads/999").status_code == 404


def test_create_post_returns_201_and_counts(client):
    thread_id = _thread(client).get_json()["id"]
    resp = _post(client, thread_id, "Bilal", "Use f-strings.")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "thread_id": thread_id,
        "author": "Bilal",
        "content": "Use f-strings.",
        "upvotes": 0,
    }
    assert client.get(f"/threads/{thread_id}").get_json()["post_count"] == 1


def test_create_post_requires_author_and_content(client):
    thread_id = _thread(client).get_json()["id"]
    assert _post(client, thread_id, "", "Hi").status_code == 400
    assert _post(client, thread_id, "Usman", "   ").status_code == 400
    assert _post(client, 999, "Usman", "Hi").status_code == 404


def test_list_posts_sorted_by_upvotes(client):
    thread_id = _thread(client).get_json()["id"]
    first = _post(client, thread_id, content="First").get_json()["id"]
    second = _post(client, thread_id, content="Second").get_json()["id"]
    client.post(f"/posts/{second}/upvote")
    client.post(f"/posts/{second}/upvote")
    client.post(f"/posts/{first}/upvote")
    listed = client.get(f"/threads/{thread_id}/posts").get_json()
    assert [p["id"] for p in listed] == [second, first]
    assert [p["upvotes"] for p in listed] == [2, 1]


def test_upvote_increments(client):
    thread_id = _thread(client).get_json()["id"]
    post_id = _post(client, thread_id).get_json()["id"]
    for expected in (1, 2, 3):
        resp = client.post(f"/posts/{post_id}/upvote")
        assert resp.status_code == 200
        assert resp.get_json()["upvotes"] == expected


def test_upvote_unknown_post_is_404(client):
    assert client.post("/posts/999/upvote").status_code == 404
    assert client.get("/posts/999").status_code == 404


def test_get_post(client):
    thread_id = _thread(client).get_json()["id"]
    post_id = _post(client, thread_id, "Bilal", "Hello").get_json()["id"]
    assert client.get(f"/posts/{post_id}").get_json()["author"] == "Bilal"


def test_search_threads_by_keyword(client):
    _thread(client, "Best Python tips", "Usman")
    _thread(client, "Flask vs Django in 2026", "Ayesha")
    found = client.get("/threads", query_string={"q": "flask"}).get_json()
    assert len(found) == 1
    assert found[0]["title"] == "Flask vs Django in 2026"
    # Search is case-insensitive and matches substrings.
    assert len(client.get("/threads", query_string={"q": "PYTHON"}).get_json()) == 1
    assert client.get("/threads", query_string={"q": "rust"}).get_json() == []
    assert len(client.get("/threads").get_json()) == 2


def test_close_thread(client):
    thread_id = _thread(client).get_json()["id"]
    closed = client.post(f"/threads/{thread_id}/close")
    assert closed.status_code == 200
    assert closed.get_json()["status"] == "closed"


def test_close_unknown_thread_is_404(client):
    assert client.post("/threads/999/close").status_code == 404


def test_close_twice_returns_409(client):
    thread_id = _thread(client).get_json()["id"]
    client.post(f"/threads/{thread_id}/close")
    again = client.post(f"/threads/{thread_id}/close")
    assert again.status_code == 409
    assert "already" in again.get_json()["error"]


def test_post_to_closed_thread_returns_409(client):
    thread_id = _thread(client).get_json()["id"]
    client.post(f"/threads/{thread_id}/close")
    resp = _post(client, thread_id, "Usman", "Too late.")
    assert resp.status_code == 409
    assert "closed" in resp.get_json()["error"]


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
