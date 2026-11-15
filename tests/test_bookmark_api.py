"""Tests for bookmark_api."""
import pytest

from bookmark_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _create(client, title="Flask docs", url="https://flask.palletsprojects.com/",
            tags=None):
    payload = {"title": title, "url": url}
    if tags is not None:
        payload["tags"] = tags
    return client.post("/bookmarks", json=payload)


def test_create_bookmark(client):
    resp = _create(client, tags=["python", "web"])
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["title"] == "Flask docs"
    assert data["tags"] == ["python", "web"]


def test_create_defaults_tags_to_empty(client):
    data = _create(client).get_json()
    assert data["tags"] == []


def test_list_bookmarks(client):
    _create(client)
    _create(client, title="Real Python", url="https://realpython.com/")
    listed = client.get("/bookmarks").get_json()
    assert len(listed) == 2
    assert [b["id"] for b in listed] == [1, 2]


def test_filter_by_tag(client):
    _create(client, tags=["python"])
    _create(client, title="MDN", url="https://developer.mozilla.org/", tags=["web"])
    tagged = client.get("/bookmarks?tag=python").get_json()
    assert len(tagged) == 1
    assert tagged[0]["title"] == "Flask docs"


def test_search_by_title(client):
    _create(client)
    _create(client, title="Real Python", url="https://realpython.com/")
    found = client.get("/bookmarks?q=FLASK").get_json()
    assert len(found) == 1
    assert found[0]["title"] == "Flask docs"


def test_get_one_bookmark(client):
    created = _create(client).get_json()
    fetched = client.get(f"/bookmarks/{created['id']}").get_json()
    assert fetched["url"] == "https://flask.palletsprojects.com/"


def test_get_missing_bookmark_404(client):
    resp = client.get("/bookmarks/99")
    assert resp.status_code == 404
    assert "not found" in resp.get_json()["error"]


def test_create_rejects_bad_url(client):
    for bad in ("not-a-url", "ftp://example.com/x", "https://", ""):
        resp = _create(client, url=bad)
        assert resp.status_code == 400, bad
        assert "url" in resp.get_json()["error"]


def test_create_rejects_missing_title(client):
    resp = client.post("/bookmarks", json={"url": "https://example.com/"})
    assert resp.status_code == 400
    assert "title" in resp.get_json()["error"]


def test_create_rejects_bad_tags(client):
    resp = _create(client, tags="python")
    assert resp.status_code == 400
    resp = _create(client, tags=["ok", ""])
    assert resp.status_code == 400


def test_update_bookmark(client):
    created = _create(client).get_json()
    resp = client.put(
        f"/bookmarks/{created['id']}",
        json={"title": "Flask", "tags": ["docs"]},
    )
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Flask"
    assert resp.get_json()["tags"] == ["docs"]


def test_update_rejects_bad_url(client):
    created = _create(client).get_json()
    resp = client.put(f"/bookmarks/{created['id']}", json={"url": "nope"})
    assert resp.status_code == 400


def test_update_missing_bookmark_404(client):
    resp = client.put("/bookmarks/99", json={"title": "x"})
    assert resp.status_code == 404


def test_delete_bookmark(client):
    created = _create(client).get_json()
    resp = client.delete(f"/bookmarks/{created['id']}")
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] == created["id"]
    assert client.get("/bookmarks").get_json() == []


def test_delete_missing_bookmark_404(client):
    assert client.delete("/bookmarks/99").status_code == 404


def test_unknown_route_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
