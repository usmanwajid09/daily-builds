"""REST household-chore tracker API built with Flask.

In-memory store, fresh for each app instance. Chores rotate between
assignees: marking a chore done moves the turn to the next person, and a
chore is flagged overdue when its due date (last done + frequency) has
passed.

Endpoints:
- ``GET /chores`` — list all chores (with current assignee + overdue flag)
- ``POST /chores`` — create a chore (``{"name", "assignees": [names],
  "frequency_days": n?}``)
- ``GET /chores/<id>`` — fetch one chore
- ``POST /chores/<id>/done`` — mark done; rotates to the next assignee
  (optional ``{"date": "YYYY-MM-DD"}``, defaults to today)
- ``PUT /chores/<id>`` — update name / assignees / frequency
- ``DELETE /chores/<id>`` — delete a chore
"""
import re
from datetime import date, timedelta

from flask import Flask, jsonify, request

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def create_app():
    """Return a Flask app with a fresh, empty in-memory chore store."""
    app = Flask(__name__)
    app.config["CHORES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _valid_date(value):
        if not isinstance(value, str) or not _DATE_RE.match(value):
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True

    def _valid_assignees(value):
        return (
            isinstance(value, list)
            and len(value) >= 1
            and all(isinstance(a, str) and a.strip() for a in value)
        )

    def _public(chore, today=None):
        today = today or date.today()
        if chore["last_done"] is None:
            due = chore["created"]
            overdue = today > date.fromisoformat(due) + timedelta(
                days=chore["frequency_days"]
            )
        else:
            due = (
                date.fromisoformat(chore["last_done"])
                + timedelta(days=chore["frequency_days"])
            ).isoformat()
            overdue = today > date.fromisoformat(due)
        return {
            "id": chore["id"],
            "name": chore["name"],
            "assignees": list(chore["assignees"]),
            "current_assignee": chore["assignees"][chore["turn_index"]],
            "frequency_days": chore["frequency_days"],
            "last_done": chore["last_done"],
            "due_date": due,
            "overdue": overdue,
            "times_done": chore["times_done"],
        }

    def _get_chore(chore_id):
        chore = app.config["CHORES"].get(chore_id)
        if chore is None:
            return None, _error(f"chore {chore_id} not found", 404)
        return chore, None

    @app.get("/chores")
    def list_chores():
        chores = sorted(app.config["CHORES"].values(), key=lambda c: c["id"])
        return jsonify([_public(c) for c in chores])

    @app.post("/chores")
    def create_chore():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        assignees = data.get("assignees")
        if not _valid_assignees(assignees):
            return _error("assignees is required and must be a non-empty list of names", 400)
        frequency = data.get("frequency_days", 7)
        if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency < 1:
            return _error("frequency_days must be a positive integer", 400)
        chore_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        chore = {
            "id": chore_id,
            "name": name.strip(),
            "assignees": [a.strip() for a in assignees],
            "turn_index": 0,
            "frequency_days": frequency,
            "last_done": None,
            "created": date.today().isoformat(),
            "times_done": 0,
        }
        app.config["CHORES"][chore_id] = chore
        return jsonify(_public(chore)), 201

    @app.get("/chores/<int:chore_id>")
    def get_chore(chore_id):
        chore, err = _get_chore(chore_id)
        if err is not None:
            return err
        return jsonify(_public(chore))

    @app.post("/chores/<int:chore_id>/done")
    def mark_done(chore_id):
        chore, err = _get_chore(chore_id)
        if err is not None:
            return err
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        done_date = data.get("date", date.today().isoformat())
        if not _valid_date(done_date):
            return _error("date must be a valid YYYY-MM-DD date", 400)
        chore["last_done"] = done_date
        chore["times_done"] += 1
        # Auto-rotate to the next assignee.
        chore["turn_index"] = (chore["turn_index"] + 1) % len(chore["assignees"])
        return jsonify(_public(chore))

    @app.put("/chores/<int:chore_id>")
    def update_chore(chore_id):
        chore, err = _get_chore(chore_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "name" in data:
            if not isinstance(data["name"], str) or not data["name"].strip():
                return _error("name must be a non-empty string", 400)
            chore["name"] = data["name"].strip()
        if "assignees" in data:
            if not _valid_assignees(data["assignees"]):
                return _error("assignees must be a non-empty list of names", 400)
            chore["assignees"] = [a.strip() for a in data["assignees"]]
            chore["turn_index"] = 0
        if "frequency_days" in data:
            freq = data["frequency_days"]
            if not isinstance(freq, int) or isinstance(freq, bool) or freq < 1:
                return _error("frequency_days must be a positive integer", 400)
            chore["frequency_days"] = freq
        return jsonify(_public(chore))

    @app.delete("/chores/<int:chore_id>")
    def delete_chore(chore_id):
        chore, err = _get_chore(chore_id)
        if err is not None:
            return err
        del app.config["CHORES"][chore_id]
        return jsonify({"deleted": chore_id})

    return app
