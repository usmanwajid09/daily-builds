"""Tests for the url_shortener Flask app, using Flask's test client."""
import pytest

from url_shortener.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _shorten(client, url="https://example.com/page"):
    resp = client.post("/shorten", json={"url": url})
    assert resp.status_code == 201
    return resp.get_json()


def test_index_describes_service(client):
    data = client.get("/").get_json()
    assert data["service"] == "url_shortener"


def test_shorten_returns_code_and_short_url(client):
    data = _shorten(client)
    assert len(data["code"]) == 6
    assert data["short_url"].endswith("/" + data["code"])
    assert data["url"] == "https://example.com/page"


def test_shorten_gives_unique_codes(client):
    first = _shorten(client, "https://example.com/a")
    second = _shorten(client, "https://example.com/b")
    assert first["code"] != second["code"]


def test_shorten_rejects_bad_urls(client):
    for bad in ("", "not-a-url", "ftp://example.com", "https://", 42, None):
        resp = client.post("/shorten", json={"url": bad})
        assert resp.status_code == 400, bad
        assert "url" in resp.get_json()["error"]


def test_shorten_rejects_missing_and_non_json_body(client):
    assert client.post("/shorten", json={}).status_code == 400
    resp = client.post("/shorten", data="x", content_type="text/plain")
    assert resp.status_code == 400


def test_redirect_goes_to_original_url(client):
    code = _shorten(client)["code"]
    resp = client.get(f"/{code}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"] == "https://example.com/page"


def test_clicks_are_counted(client):
    code = _shorten(client)["code"]
    assert client.get(f"/stats/{code}").get_json()["clicks"] == 0
    client.get(f"/{code}")
    client.get(f"/{code}")
    stats = client.get(f"/stats/{code}").get_json()
    assert stats["clicks"] == 2
    assert stats["url"] == "https://example.com/page"


def test_unknown_code_is_404(client):
    resp = client.get("/zzzzzz", follow_redirects=False)
    assert resp.status_code == 404
    assert "not found" in resp.get_json()["error"]
    resp = client.get("/stats/zzzzzz")
    assert resp.status_code == 404
    assert "not found" in resp.get_json()["error"]
