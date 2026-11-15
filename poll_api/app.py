"""REST polls/voting API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /polls`` — list all polls (question, options, vote counts)
- ``POST /polls`` — create a poll
  (``{"question": str, "options": [str, str, ...]}``, at least 2 options)
- ``GET /polls/<id>`` — fetch one poll with its current results
- ``POST /polls/<id>/vote`` — vote once per voter
  (``{"voter_id": str, "option": str}``)
- ``GET /polls/<id>/results`` — counts and percentages per option
- ``DELETE /polls/<id>`` — delete a poll
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory poll store."""
    app = Flask(__name__)
    app.config["POLLS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_poll(poll_id):
        poll = app.config["POLLS"].get(poll_id)
        if poll is None:
            return None, _error(f"poll {poll_id} not found", 404)
        return poll, None

    def _public_poll(poll):
        counts = {opt: 0 for opt in poll["options"]}
        for voted_option in poll["votes"].values():
            counts[voted_option] += 1
        return {
            "id": poll["id"],
            "question": poll["question"],
            "options": list(poll["options"]),
            "votes": counts,
            "total_votes": len(poll["votes"]),
        }

    def _results(poll):
        counts = {opt: 0 for opt in poll["options"]}
        for voted_option in poll["votes"].values():
            counts[voted_option] += 1
        total = len(poll["votes"])
        results = []
        for opt in poll["options"]:
            count = counts[opt]
            percentage = round(100.0 * count / total, 1) if total else 0.0
            results.append(
                {"option": opt, "votes": count, "percentage": percentage}
            )
        return {
            "id": poll["id"],
            "question": poll["question"],
            "total_votes": total,
            "results": results,
        }

    @app.get("/polls")
    def list_polls():
        polls = sorted(app.config["POLLS"].values(), key=lambda p: p["id"])
        return jsonify([_public_poll(p) for p in polls])

    @app.post("/polls")
    def create_poll():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            return _error("question is required and must be a non-empty string", 400)
        options = data.get("options")
        if not isinstance(options, list) or len(options) < 2:
            return _error("options must be a list of at least 2 strings", 400)
        cleaned = []
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                return _error("each option must be a non-empty string", 400)
            cleaned.append(opt.strip())
        if len(set(cleaned)) != len(cleaned):
            return _error("options must be unique", 400)
        poll_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        poll = {
            "id": poll_id,
            "question": question.strip(),
            "options": cleaned,
            "votes": {},
        }
        app.config["POLLS"][poll_id] = poll
        return jsonify(_public_poll(poll)), 201

    @app.get("/polls/<int:poll_id>")
    def get_poll(poll_id):
        poll, err = _get_poll(poll_id)
        if err is not None:
            return err
        return jsonify(_public_poll(poll))

    @app.post("/polls/<int:poll_id>/vote")
    def vote(poll_id):
        poll, err = _get_poll(poll_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        voter_id = data.get("voter_id")
        if not isinstance(voter_id, str) or not voter_id.strip():
            return _error("voter_id is required and must be a non-empty string", 400)
        option = data.get("option")
        if not isinstance(option, str) or option not in poll["options"]:
            return _error(
                f"option must be one of: {', '.join(poll['options'])}", 400
            )
        voter_id = voter_id.strip()
        if voter_id in poll["votes"]:
            return _error(f"voter {voter_id} has already voted in this poll", 409)
        poll["votes"][voter_id] = option
        return jsonify(_results(poll)), 201

    @app.get("/polls/<int:poll_id>/results")
    def get_results(poll_id):
        poll, err = _get_poll(poll_id)
        if err is not None:
            return err
        return jsonify(_results(poll))

    @app.delete("/polls/<int:poll_id>")
    def delete_poll(poll_id):
        _poll, err = _get_poll(poll_id)
        if err is not None:
            return err
        del app.config["POLLS"][poll_id]
        return jsonify({"deleted": poll_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
