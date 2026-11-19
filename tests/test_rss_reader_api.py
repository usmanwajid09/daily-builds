"""Tests for the rss_reader_api Flask app, using Flask's test client."""
import pytest

from rss_reader_api.app import create_app

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Daily Dev News</title>
    <link>https://example.com/news</link>
    <description>Short dev headlines.</description>
    <item>
      <title>Flask 3.1 released</title>
      <link>https://example.com/news/flask-31</link>
      <description>Async views and more.</description>
      <pubDate>Mon, 17 Nov 2026 09:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Python 3.14 lands</title>
      <link>https://example.com/news/py314</link>
      <description>Free-threaded by default.</description>
      <pubDate>Tue, 18 Nov 2026 09:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _add_feed(client, name="dev-news", xml=SAMPLE_XML):
    return client.post("/feeds", json={"name": name, "xml": xml})


def test_list_feeds_starts_empty(client):
    assert client.get("/feeds").get_json() == []


def test_add_feed_returns_201(client):
    resp = _add_feed(client)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "dev-news",
        "title": "Daily Dev News",
        "item_count": 2,
    }


def test_get_feed_details(client):
    created = _add_feed(client).get_json()
    fetched = client.get(f"/feeds/{created['id']}").get_json()
    assert fetched == created


def test_get_items(client):
    created = _add_feed(client).get_json()
    items = client.get(f"/feeds/{created['id']}/items").get_json()
    assert len(items) == 2
    assert items[0]["title"] == "Flask 3.1 released"
    assert items[0]["link"] == "https://example.com/news/flask-31"
    assert items[0]["description"] == "Async views and more."
    assert items[0]["pubDate"] == "Mon, 17 Nov 2026 09:00:00 GMT"
    assert items[1]["title"] == "Python 3.14 lands"


def test_add_feed_rejects_malformed_xml(client):
    resp = _add_feed(client, xml="<rss><oops>")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_add_feed_rejects_non_rss_root(client):
    resp = _add_feed(client, xml="<html><body>hi</body></html>")
    assert resp.status_code == 400


def test_add_feed_rejects_missing_channel(client):
    resp = _add_feed(client, xml="<rss version='2.0'></rss>")
    assert resp.status_code == 400


def test_add_feed_rejects_empty_name(client):
    assert _add_feed(client, name="   ").status_code == 400


def test_add_feed_rejects_empty_xml(client):
    assert _add_feed(client, xml="").status_code == 400


def test_add_feed_rejects_non_json_body(client):
    assert client.post("/feeds", data="not json").status_code == 400


def test_feed_with_no_items(client):
    xml = "<rss version='2.0'><channel><title>Empty</title></channel></rss>"
    created = _add_feed(client, xml=xml).get_json()
    assert created["item_count"] == 0
    assert created["title"] == "Empty"
    assert client.get(f"/feeds/{created['id']}/items").get_json() == []


def test_item_missing_fields_default_empty(client):
    xml = (
        "<rss version='2.0'><channel><title>T</title>"
        "<item><title>Only title</title></item></channel></rss>"
    )
    created = _add_feed(client, xml=xml).get_json()
    item = client.get(f"/feeds/{created['id']}/items").get_json()[0]
    assert item == {
        "title": "Only title",
        "link": "",
        "description": "",
        "pubDate": "",
    }


def test_get_unknown_feed_404(client):
    assert client.get("/feeds/99").status_code == 404
    assert client.get("/feeds/99/items").status_code == 404


def test_delete_feed(client):
    created = _add_feed(client).get_json()
    assert client.delete(f"/feeds/{created['id']}").status_code == 200
    assert client.get("/feeds").get_json() == []
    assert client.delete(f"/feeds/{created['id']}").status_code == 404


def test_list_multiple_feeds(client):
    _add_feed(client, name="one")
    _add_feed(client, name="two")
    feeds = client.get("/feeds").get_json()
    assert [f["name"] for f in feeds] == ["one", "two"]
