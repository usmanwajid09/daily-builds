"""Flask notes API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /notes`` — create a note (``{"title": str, "body": str?,
  "tags": [str]?}``)
- ``GET /notes`` — list notes; ``?search=<text>`` matches title or body
  (case-insensitive), ``?tag=<tag>`` filters by tag
- ``GET /notes/<id>`` — fetch one note
- ``PUT /notes/<id>`` — update title, body and/or tags
- ``DELETE /notes/<id>`` — delete a note
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory notes store."""
    app = Flask(__name__)
    app.config["NOTES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_note(note_id):
        note = app.config["NOTES"].get(note_id)
        if note is None:
            return None, _error(f"note {note_id} not found", 404)
        return note, None

    def _validate_tags(tags):
        if not isinstance(tags, list):
            return "tags must be a list of strings"
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                return "each tag must be a non-empty string"
        return None

    def _clean_tags(tags):
        seen = set()
        cleaned = []
        for tag in tags:
            tag = tag.strip()
            if tag not in seen:
                seen.add(tag)
                cleaned.append(tag)
        return cleaned

    @app.post("/notes")
    def create_note():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        body = data.get("body", "")
        if not isinstance(body, str):
            return _error("body must be a string", 400)
        tags = data.get("tags", [])
        tag_error = _validate_tags(tags)
        if tag_error:
            return _error(tag_error, 400)
        note_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        note = {
            "id": note_id,
            "title": title.strip(),
            "body": body,
            "tags": _clean_tags(tags),
        }
        app.config["NOTES"][note_id] = note
        return jsonify(note), 201

    @app.get("/notes")
    def list_notes():
        notes = sorted(app.config["NOTES"].values(), key=lambda n: n["id"])
        search = request.args.get("search")
        if search:
            needle = search.lower()
            notes = [
                n for n in notes
                if needle in n["title"].lower() or needle in n["body"].lower()
            ]
        tag = request.args.get("tag")
        if tag:
            notes = [n for n in notes if tag in n["tags"]]
        return jsonify(notes)

    @app.get("/notes/<int:note_id>")
    def get_note(note_id):
        note, err = _get_note(note_id)
        if err is not None:
            return err
        return jsonify(note)

    @app.put("/notes/<int:note_id>")
    def update_note(note_id):
        note, err = _get_note(note_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if not any(k in data for k in ("title", "body", "tags")):
            return _error("nothing to update: provide title, body and/or tags", 400)
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            note["title"] = data["title"].strip()
        if "body" in data:
            if not isinstance(data["body"], str):
                return _error("body must be a string", 400)
            note["body"] = data["body"]
        if "tags" in data:
            tag_error = _validate_tags(data["tags"])
            if tag_error:
                return _error(tag_error, 400)
            note["tags"] = _clean_tags(data["tags"])
        return jsonify(note)

    @app.delete("/notes/<int:note_id>")
    def delete_note(note_id):
        _note, err = _get_note(note_id)
        if err is not None:
            return err
        del app.config["NOTES"][note_id]
        return jsonify({"deleted": note_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
