"""REST URL health-checker API built with Flask.

Checks a URL with the standard-library ``urllib`` (a HEAD request,
falling back to GET) and reports the HTTP status code and response
time. Unreachable hosts return a clean 502-style JSON error instead
of crashing. Check history lives in an in-memory store, fresh for
each app instance. All responses are JSON. Endpoints:

- ``POST /check`` — check a URL (``{"url": str}``)
- ``GET /checks`` — list past checks
- ``GET /checks/<id>`` — one past check
"""
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from flask import Flask, jsonify, request


def _check_url(url):
    """Return (result_dict, error_message or None) for one URL.

    Assumes the URL has already passed scheme/host validation; a
    failure here means the host could not be reached.
    """
    request_obj = urllib.request.Request(url, method="HEAD")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request_obj, timeout=5) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        # A real HTTP response (e.g. 404/500) still counts as reachable.
        status = exc.code
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, f"could not reach {url}: {exc.reason if hasattr(exc, 'reason') else exc}"
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    return {
        "url": url,
        "status": status,
        "ok": 200 <= status < 400,
        "response_time_ms": elapsed_ms,
    }, None


def create_app():
    """Return a Flask app with a fresh, empty in-memory check store."""
    app = Flask(__name__)
    app.config["CHECKS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    @app.post("/check")
    def check():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        url = data.get("url")
        if not isinstance(url, str) or not url.strip():
            return _error("url is required and must be a non-empty string", 400)
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return _error("url must be an http(s) URL with a host", 400)
        result, error = _check_url(url)
        if error is not None:
            return _error(error, 502)
        check_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        record = {"id": check_id, **result}
        app.config["CHECKS"][check_id] = record
        return jsonify(record), 200

    @app.get("/checks")
    def list_checks():
        checks = sorted(app.config["CHECKS"].values(), key=lambda c: c["id"])
        return jsonify(checks)

    @app.get("/checks/<int:check_id>")
    def get_check(check_id):
        record = app.config["CHECKS"].get(check_id)
        if record is None:
            return _error(f"check {check_id} not found", 404)
        return jsonify(record)

    return app
