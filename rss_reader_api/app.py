"""REST RSS-reader API built with Flask.

Feeds are submitted as raw RSS XML strings and parsed with the
standard-library ``xml`` module — no extra dependency. Parsed feeds
live in an in-memory store, fresh for each app instance. All
responses are JSON. Endpoints:

- ``GET /feeds`` — list all feeds (id, name, title, item count)
- ``POST /feeds`` — add a feed (``{"name": str, "xml": str}``)
- ``GET /feeds/<id>`` — one feed's details
- ``GET /feeds/<id>/items`` — the feed's items (title, link,
  description, pubDate)
- ``DELETE /feeds/<id>`` — remove a feed
"""
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, request


def _parse_rss(xml_text):
    """Parse an RSS 2.0 XML string into (channel title, [items]).

    Raises ValueError with a human-readable message on bad input.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"invalid RSS XML: {exc}") from exc
    if root.tag != "rss":
        raise ValueError("invalid RSS XML: root element must be <rss>")
    channel = root.find("channel")
    if channel is None:
        raise ValueError("invalid RSS XML: no <channel> element found")
    title_el = channel.find("title")
    title = title_el.text.strip() if title_el is not None and title_el.text else ""
    items = []
    for item_el in channel.findall("item"):
        def _text(tag):
            el = item_el.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        items.append(
            {
                "title": _text("title"),
                "link": _text("link"),
                "description": _text("description"),
                "pubDate": _text("pubDate"),
            }
        )
    return title, items


def create_app():
    """Return a Flask app with a fresh, empty in-memory feed store."""
    app = Flask(__name__)
    app.config["FEEDS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_feed(feed_id):
        feed = app.config["FEEDS"].get(feed_id)
        if feed is None:
            return None, _error(f"feed {feed_id} not found", 404)
        return feed, None

    @app.get("/feeds")
    def list_feeds():
        feeds = [
            {
                "id": f["id"],
                "name": f["name"],
                "title": f["title"],
                "item_count": len(f["items"]),
            }
            for f in sorted(app.config["FEEDS"].values(), key=lambda f: f["id"])
        ]
        return jsonify(feeds)

    @app.post("/feeds")
    def add_feed():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        xml_text = data.get("xml")
        if not isinstance(xml_text, str) or not xml_text.strip():
            return _error("xml is required and must be a non-empty string", 400)
        try:
            title, items = _parse_rss(xml_text)
        except ValueError as exc:
            return _error(str(exc), 400)
        feed_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        feed = {"id": feed_id, "name": name.strip(), "title": title, "items": items}
        app.config["FEEDS"][feed_id] = feed
        return jsonify(
            {
                "id": feed["id"],
                "name": feed["name"],
                "title": feed["title"],
                "item_count": len(items),
            }
        ), 201

    @app.get("/feeds/<int:feed_id>")
    def get_feed(feed_id):
        feed, err = _get_feed(feed_id)
        if err is not None:
            return err
        return jsonify(
            {
                "id": feed["id"],
                "name": feed["name"],
                "title": feed["title"],
                "item_count": len(feed["items"]),
            }
        )

    @app.get("/feeds/<int:feed_id>/items")
    def get_items(feed_id):
        feed, err = _get_feed(feed_id)
        if err is not None:
            return err
        return jsonify(feed["items"])

    @app.delete("/feeds/<int:feed_id>")
    def delete_feed(feed_id):
        if feed_id not in app.config["FEEDS"]:
            return _error(f"feed {feed_id} not found", 404)
        del app.config["FEEDS"][feed_id]
        return jsonify({"deleted": feed_id})

    return app
