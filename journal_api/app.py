"""Flask personal journal API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /entries`` — create an entry (``{"date": "YYYY-MM-DD",
  "mood": 1-5, "text": str, "tags": [str]?}``)
- ``GET /entries`` — list entries, optionally filtered by
  ``?from=YYYY-MM-DD&to=YYYY-MM-DD``
- ``GET /entries/<id>`` — fetch one entry
- ``DELETE /entries/<id>`` — delete an entry
- ``GET /entries/search?tag=<tag>`` — entries carrying a tag
  (case-insensitive)
- ``GET /stats/mood-average`` — average mood over an optional
  ``?from=...&to=...`` range (``{"count": int, "average_mood": float}``)

Mood must be an integer from 1 to 5 and dates must be valid
``YYYY-MM-DD`` strings, otherwise the request returns ``400``.
Unknown entries return ``404``.
"""
from datetime import date

from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory journal store."""
    app = Flask(__name__)
    app.config["ENTRIES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_date(value):
        """Return a date object for a YYYY-MM-DD string, else None."""
        if not isinstance(value, str):
            return None
        try:
            parts = value.split("-")
            if len(parts) != 3 or len(parts[0]) != 4:
                return None
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, TypeError):
            return None

    def _get_entry(entry_id):
        entry = app.config["ENTRIES"].get(entry_id)
        if entry is None:
            return None, _error(f"entry {entry_id} not found", 404)
        return entry, None

    def _in_range(entry, start, end):
        day = _parse_date(entry["date"])
        if start is not None and day < start:
            return False
        if end is not None and day > end:
            return False
        return True

    def _range_params():
        """Parse ?from/&to query params; return (start, end, error)."""
        start = _parse_date(request.args.get("from")) if request.args.get("from") else None
        end = _parse_date(request.args.get("to")) if request.args.get("to") else None
        if request.args.get("from") and start is None:
            return None, None, _error("from must be a valid YYYY-MM-DD date", 400)
        if request.args.get("to") and end is None:
            return None, None, _error("to must be a valid YYYY-MM-DD date", 400)
        if start is not None and end is not None and end < start:
            return None, None, _error("to cannot be before from", 400)
        return start, end, None

    def _public(entry):
        return {
            "id": entry["id"],
            "date": entry["date"],
            "mood": entry["mood"],
            "text": entry["text"],
            "tags": list(entry["tags"]),
        }

    @app.post("/entries")
    def create_entry():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if _parse_date(data.get("date")) is None:
            return _error("date is required and must be a valid YYYY-MM-DD string", 400)
        mood = data.get("mood")
        if isinstance(mood, bool) or not isinstance(mood, int) or not 1 <= mood <= 5:
            return _error("mood is required and must be an integer from 1 to 5", 400)
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return _error("text is required and must be a non-empty string", 400)
        tags = data.get("tags", [])
        if not isinstance(tags, list) or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            return _error("tags must be a list of non-empty strings", 400)
        entry_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        entry = {
            "id": entry_id,
            "date": data["date"],
            "mood": mood,
            "text": text.strip(),
            "tags": [tag.strip() for tag in tags],
        }
        app.config["ENTRIES"][entry_id] = entry
        return jsonify(_public(entry)), 201

    @app.get("/entries")
    def list_entries():
        start, end, err = _range_params()
        if err is not None:
            return err
        entries = sorted(app.config["ENTRIES"].values(), key=lambda e: (e["date"], e["id"]))
        entries = [e for e in entries if _in_range(e, start, end)]
        return jsonify([_public(e) for e in entries])

    @app.get("/entries/search")
    def search_entries():
        tag = request.args.get("tag", "").strip().lower()
        if not tag:
            return _error("tag query parameter is required", 400)
        entries = sorted(app.config["ENTRIES"].values(), key=lambda e: (e["date"], e["id"]))
        matches = [
            e for e in entries if any(t.lower() == tag for t in e["tags"])
        ]
        return jsonify([_public(e) for e in matches])

    @app.get("/entries/<int:entry_id>")
    def get_entry(entry_id):
        entry, err = _get_entry(entry_id)
        if err is not None:
            return err
        return jsonify(_public(entry))

    @app.delete("/entries/<int:entry_id>")
    def delete_entry(entry_id):
        _entry, err = _get_entry(entry_id)
        if err is not None:
            return err
        del app.config["ENTRIES"][entry_id]
        return jsonify({"deleted": entry_id})

    @app.get("/stats/mood-average")
    def mood_average():
        start, end, err = _range_params()
        if err is not None:
            return err
        entries = [e for e in app.config["ENTRIES"].values() if _in_range(e, start, end)]
        if not entries:
            return jsonify({"count": 0, "average_mood": None})
        average = round(sum(e["mood"] for e in entries) / len(entries), 2)
        return jsonify({"count": len(entries), "average_mood": average})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
