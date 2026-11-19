"""REST snippet-sharing (pastebin) API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /snippets`` — list all live (non-expired) snippets
- ``POST /snippets`` — create one (``{"content": str, "expires_in": int?}``,
  where ``expires_in`` is seconds until expiry)
- ``GET /snippets/<id>`` — fetch one snippet; expired ones return 410 Gone
- ``DELETE /snippets/<id>`` — delete a snippet
"""
import time

from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory snippet store."""
    app = Flask(__name__)
    app.config["SNIPPETS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_expired(snippet):
        return snippet["expires_at"] is not None and time.time() > snippet["expires_at"]

    def _public(snippet):
        return {
            "id": snippet["id"],
            "content": snippet["content"],
            "created_at": snippet["created_at"],
            "expires_at": snippet["expires_at"],
        }

    @app.get("/snippets")
    def list_snippets():
        live = [
            _public(s)
            for s in sorted(app.config["SNIPPETS"].values(), key=lambda s: s["id"])
            if not _is_expired(s)
        ]
        return jsonify(live)

    @app.post("/snippets")
    def create_snippet():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            return _error("content is required and must be a non-empty string", 400)
        expires_in = data.get("expires_in")
        if expires_in is not None:
            if not isinstance(expires_in, int) or isinstance(expires_in, bool):
                return _error("expires_in must be an integer number of seconds", 400)
            if expires_in <= 0:
                return _error("expires_in must be a positive number of seconds", 400)
        now = time.time()
        snippet_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        snippet = {
            "id": snippet_id,
            "content": content,
            "created_at": now,
            "expires_at": now + expires_in if expires_in is not None else None,
        }
        app.config["SNIPPETS"][snippet_id] = snippet
        return jsonify(_public(snippet)), 201

    @app.get("/snippets/<int:snippet_id>")
    def get_snippet(snippet_id):
        snippet = app.config["SNIPPETS"].get(snippet_id)
        if snippet is None:
            return _error(f"snippet {snippet_id} not found", 404)
        if _is_expired(snippet):
            return _error(f"snippet {snippet_id} has expired", 410)
        return jsonify(_public(snippet))

    @app.delete("/snippets/<int:snippet_id>")
    def delete_snippet(snippet_id):
        if snippet_id not in app.config["SNIPPETS"]:
            return _error(f"snippet {snippet_id} not found", 404)
        del app.config["SNIPPETS"][snippet_id]
        return jsonify({"deleted": snippet_id})

    return app
