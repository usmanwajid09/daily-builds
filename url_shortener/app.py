"""URL shortener built with Flask.

In-memory store, fresh for each app instance. Endpoints:

- ``POST /shorten`` — ``{"url": "https://..."}`` returns a short code
- ``GET /<code>`` — redirects (302) to the original URL, counts a click
- ``GET /stats/<code>`` — ``{"code", "url", "clicks"}``
- ``GET /`` — service info

Short codes are base-62 encodings of an incrementing counter, so they
are unique without any randomness or collision handling.
"""
from urllib.parse import urlparse

from flask import Flask, jsonify, redirect, request

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _encode(n):
    """Encode a positive integer in base 62, left-padded to 6 chars."""
    if n <= 0:
        raise ValueError("counter must be positive")
    chars = []
    while n:
        n, rem = divmod(n, 62)
        chars.append(_ALPHABET[rem])
    return "".join(reversed(chars)).rjust(6, "0")


def _is_valid_url(url):
    """A URL is valid if it is an http(s) URL with a host name."""
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def create_app():
    """Return a Flask app with a fresh, empty in-memory link store."""
    app = Flask(__name__)
    app.config["LINKS"] = {}
    app.config["COUNTER"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "url_shortener",
                "usage": "POST /shorten with {\"url\": \"https://example.com\"}",
            }
        )

    @app.post("/shorten")
    def shorten():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        url = data.get("url")
        if not _is_valid_url(url):
            return _error(
                "url is required and must be a valid http:// or https:// URL", 400
            )
        code = _encode(app.config["COUNTER"])
        app.config["COUNTER"] += 1
        app.config["LINKS"][code] = {"url": url.strip(), "clicks": 0}
        return (
            jsonify(
                {
                    "code": code,
                    "url": url.strip(),
                    "short_url": request.host_url + code,
                }
            ),
            201,
        )

    @app.get("/<code>")
    def follow(code):
        link = app.config["LINKS"].get(code)
        if link is None:
            return _error(f"short code '{code}' not found", 404)
        link["clicks"] += 1
        return redirect(link["url"], code=302)

    @app.get("/stats/<code>")
    def stats(code):
        link = app.config["LINKS"].get(code)
        if link is None:
            return _error(f"short code '{code}' not found", 404)
        return jsonify(
            {"code": code, "url": link["url"], "clicks": link["clicks"]}
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
