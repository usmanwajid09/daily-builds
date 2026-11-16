"""REST feedback API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /feedback`` — list all feedback (``?rating=5`` filters to one
  rating)
- ``POST /feedback`` — submit feedback
- ``GET /feedback/summary`` — count, average rating and per-rating
  distribution
- ``GET /feedback/<id>`` — fetch one feedback entry
- ``DELETE /feedback/<id>`` — delete a feedback entry

A feedback entry is ``{"id": int, "name": str?, "rating": int,
"comment": str}``. Ratings are integers from 1 to 5 inclusive.
"""
from flask import Flask, jsonify, request

MIN_RATING = 1
MAX_RATING = 5


def create_app():
    """Return a Flask app with a fresh, empty in-memory feedback store."""
    app = Flask(__name__)
    app.config["FEEDBACK"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_entry(entry_id):
        entry = app.config["FEEDBACK"].get(entry_id)
        if entry is None:
            return None, _error(f"feedback {entry_id} not found", 404)
        return entry, None

    @app.get("/feedback")
    def list_feedback():
        entries = sorted(app.config["FEEDBACK"].values(), key=lambda e: e["id"])
        rating = request.args.get("rating")
        if rating is not None:
            try:
                wanted = int(rating)
            except ValueError:
                return _error("rating must be an integer between 1 and 5", 400)
            if wanted < MIN_RATING or wanted > MAX_RATING:
                return _error("rating must be an integer between 1 and 5", 400)
            entries = [e for e in entries if e["rating"] == wanted]
        return jsonify(entries)

    @app.post("/feedback")
    def create_feedback():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        rating = data.get("rating")
        if (
            not isinstance(rating, int)
            or isinstance(rating, bool)
            or rating < MIN_RATING
            or rating > MAX_RATING
        ):
            return _error("rating is required and must be an integer between 1 and 5", 400)
        comment = data.get("comment")
        if not isinstance(comment, str) or not comment.strip():
            return _error("comment is required and must be a non-empty string", 400)
        name = data.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            return _error("name must be a non-empty string when provided", 400)
        entry_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        entry = {"id": entry_id, "rating": rating, "comment": comment.strip()}
        if name is not None:
            entry["name"] = name.strip()
        app.config["FEEDBACK"][entry_id] = entry
        return jsonify(entry), 201

    @app.get("/feedback/summary")
    def feedback_summary():
        entries = list(app.config["FEEDBACK"].values())
        count = len(entries)
        distribution = {str(r): 0 for r in range(MIN_RATING, MAX_RATING + 1)}
        for entry in entries:
            distribution[str(entry["rating"])] += 1
        average = round(sum(e["rating"] for e in entries) / count, 2) if count else None
        return jsonify(
            {
                "count": count,
                "average_rating": average,
                "distribution": distribution,
            }
        )

    @app.get("/feedback/<int:entry_id>")
    def get_feedback(entry_id):
        entry, err = _get_entry(entry_id)
        if err is not None:
            return err
        return jsonify(entry)

    @app.delete("/feedback/<int:entry_id>")
    def delete_feedback(entry_id):
        _entry, err = _get_entry(entry_id)
        if err is not None:
            return err
        del app.config["FEEDBACK"][entry_id]
        return jsonify({"deleted": entry_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
