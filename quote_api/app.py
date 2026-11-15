"""REST quote API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /quotes`` — list quotes (``?author=<name>`` and/or ``?tag=<tag>``)
- ``POST /quotes`` — add a quote
  (``{"text": str, "author": str, "tags": [str]?}``)
- ``GET /quotes/random`` — a random quote (404 when the store is empty)
- ``GET /quotes/<id>`` — fetch one quote
- ``DELETE /quotes/<id>`` — delete a quote
"""
import random

from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory quote store."""
    app = Flask(__name__)
    app.config["QUOTES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_quote(quote_id):
        quote = app.config["QUOTES"].get(quote_id)
        if quote is None:
            return None, _error(f"quote {quote_id} not found", 404)
        return quote, None

    @app.get("/quotes")
    def list_quotes():
        quotes = sorted(app.config["QUOTES"].values(), key=lambda q: q["id"])
        author = request.args.get("author")
        if author:
            needle = author.lower()
            quotes = [q for q in quotes if needle in q["author"].lower()]
        tag = request.args.get("tag")
        if tag:
            quotes = [q for q in quotes if tag in q["tags"]]
        return jsonify(quotes)

    @app.post("/quotes")
    def create_quote():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return _error("text is required and must be a non-empty string", 400)
        author = data.get("author")
        if not isinstance(author, str) or not author.strip():
            return _error("author is required and must be a non-empty string", 400)
        tags = data.get("tags", [])
        if not isinstance(tags, list) or not all(
            isinstance(t, str) and t.strip() for t in tags
        ):
            return _error("tags must be a list of non-empty strings", 400)
        quote_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        quote = {
            "id": quote_id,
            "text": text.strip(),
            "author": author.strip(),
            "tags": [t.strip() for t in tags],
        }
        app.config["QUOTES"][quote_id] = quote
        return jsonify(quote), 201

    @app.get("/quotes/random")
    def random_quote():
        quotes = list(app.config["QUOTES"].values())
        if not quotes:
            return _error("no quotes stored yet", 404)
        return jsonify(random.choice(quotes))

    @app.get("/quotes/<int:quote_id>")
    def get_quote(quote_id):
        quote, err = _get_quote(quote_id)
        if err is not None:
            return err
        return jsonify(quote)

    @app.delete("/quotes/<int:quote_id>")
    def delete_quote(quote_id):
        _quote, err = _get_quote(quote_id)
        if err is not None:
            return err
        del app.config["QUOTES"][quote_id]
        return jsonify({"deleted": quote_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
