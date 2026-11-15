"""REST bookmarks API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /bookmarks`` — list bookmarks (``?tag=<tag>`` and/or ``?q=<text>``)
- ``POST /bookmarks`` — create (``{"title": str, "url": str, "tags": [str]?}``)
- ``GET /bookmarks/<id>`` — fetch one bookmark
- ``PUT /bookmarks/<id>`` — update title, url and/or tags
- ``DELETE /bookmarks/<id>`` — delete a bookmark
"""
from urllib.parse import urlparse

from flask import Flask, jsonify, request


def _valid_url(url):
    """True if url is a non-empty http(s) URL with a host."""
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _valid_tags(tags):
    if not isinstance(tags, list):
        return False
    return all(isinstance(t, str) and t.strip() for t in tags)


def create_app():
    """Return a Flask app with a fresh, empty in-memory bookmark store."""
    app = Flask(__name__)
    app.config["BOOKMARKS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_bookmark(bookmark_id):
        bookmark = app.config["BOOKMARKS"].get(bookmark_id)
        if bookmark is None:
            return None, _error(f"bookmark {bookmark_id} not found", 404)
        return bookmark, None

    @app.get("/bookmarks")
    def list_bookmarks():
        bookmarks = sorted(
            app.config["BOOKMARKS"].values(), key=lambda b: b["id"]
        )
        tag = request.args.get("tag")
        if tag:
            bookmarks = [b for b in bookmarks if tag in b["tags"]]
        query = request.args.get("q")
        if query:
            needle = query.lower()
            bookmarks = [
                b for b in bookmarks if needle in b["title"].lower()
            ]
        return jsonify(bookmarks)

    @app.post("/bookmarks")
    def create_bookmark():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        url = data.get("url")
        if not _valid_url(url):
            return _error("url is required and must be a valid http(s) URL", 400)
        tags = data.get("tags", [])
        if not _valid_tags(tags):
            return _error("tags must be a list of non-empty strings", 400)
        bookmark_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        bookmark = {
            "id": bookmark_id,
            "title": title.strip(),
            "url": url.strip(),
            "tags": [t.strip() for t in tags],
        }
        app.config["BOOKMARKS"][bookmark_id] = bookmark
        return jsonify(bookmark), 201

    @app.get("/bookmarks/<int:bookmark_id>")
    def get_bookmark(bookmark_id):
        bookmark, err = _get_bookmark(bookmark_id)
        if err is not None:
            return err
        return jsonify(bookmark)

    @app.put("/bookmarks/<int:bookmark_id>")
    def update_bookmark(bookmark_id):
        bookmark, err = _get_bookmark(bookmark_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if not any(k in data for k in ("title", "url", "tags")):
            return _error("nothing to update: provide title, url and/or tags", 400)
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            bookmark["title"] = data["title"].strip()
        if "url" in data:
            if not _valid_url(data["url"]):
                return _error("url must be a valid http(s) URL", 400)
            bookmark["url"] = data["url"].strip()
        if "tags" in data:
            if not _valid_tags(data["tags"]):
                return _error("tags must be a list of non-empty strings", 400)
            bookmark["tags"] = [t.strip() for t in data["tags"]]
        return jsonify(bookmark)

    @app.delete("/bookmarks/<int:bookmark_id>")
    def delete_bookmark(bookmark_id):
        _bookmark, err = _get_bookmark(bookmark_id)
        if err is not None:
            return err
        del app.config["BOOKMARKS"][bookmark_id]
        return jsonify({"deleted": bookmark_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
