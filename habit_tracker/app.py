"""Flask habit tracker API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /habits`` — create a habit (``{"name": str, "description": str?}``)
- ``GET /habits`` — list all habits, each with its streak
- ``GET /habits/<id>`` — fetch one habit
- ``POST /habits/<id>/done`` — mark a day done (``{"date": "YYYY-MM-DD"}``,
  defaults to today)
- ``DELETE /habits/<id>/done`` — unmark a day (``{"date": "YYYY-MM-DD"}``)
- ``DELETE /habits/<id>`` — delete a habit

A habit's ``streak`` is the number of consecutive days ending at its most
recent done date. ``total`` is the number of days marked done.
"""
from datetime import date, timedelta

from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory habit store."""
    app = Flask(__name__)
    app.config["HABITS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_habit(habit_id):
        habit = app.config["HABITS"].get(habit_id)
        if habit is None:
            return None, _error(f"habit {habit_id} not found", 404)
        return habit, None

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

    def _streak(done_dates):
        """Consecutive done days ending at the most recent done date."""
        if not done_dates:
            return 0
        dates = sorted(done_dates)
        streak = 1
        current = dates[-1]
        for day in reversed(dates[:-1]):
            if day == current - timedelta(days=1):
                streak += 1
                current = day
            else:
                break
        return streak

    def _public(habit):
        return {
            "id": habit["id"],
            "name": habit["name"],
            "description": habit["description"],
            "done_dates": sorted(habit["done_dates"]),
            "total": len(habit["done_dates"]),
            "streak": _streak(
                {_parse_date(d) for d in habit["done_dates"]}
            ),
        }

    @app.post("/habits")
    def create_habit():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        habit_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        habit = {
            "id": habit_id,
            "name": name.strip(),
            "description": description.strip(),
            "done_dates": set(),
        }
        app.config["HABITS"][habit_id] = habit
        return jsonify(_public(habit)), 201

    @app.get("/habits")
    def list_habits():
        habits = sorted(app.config["HABITS"].values(), key=lambda h: h["id"])
        return jsonify([_public(h) for h in habits])

    @app.get("/habits/<int:habit_id>")
    def get_habit(habit_id):
        habit, err = _get_habit(habit_id)
        if err is not None:
            return err
        return jsonify(_public(habit))

    @app.post("/habits/<int:habit_id>/done")
    def mark_done(habit_id):
        habit, err = _get_habit(habit_id)
        if err is not None:
            return err
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        raw = data.get("date")
        if raw is None:
            day = date.today()
            raw = day.isoformat()
        else:
            day = _parse_date(raw)
            if day is None:
                return _error("date must be a valid YYYY-MM-DD string", 400)
        if day > date.today():
            return _error("date cannot be in the future", 400)
        if raw in habit["done_dates"]:
            return _error(f"habit already marked done on {raw}", 400)
        habit["done_dates"].add(raw)
        return jsonify(_public(habit))

    @app.delete("/habits/<int:habit_id>/done")
    def unmark_done(habit_id):
        habit, err = _get_habit(habit_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        raw = data.get("date")
        day = _parse_date(raw)
        if day is None:
            return _error("date is required and must be a valid YYYY-MM-DD string", 400)
        if raw not in habit["done_dates"]:
            return _error(f"habit was not marked done on {raw}", 400)
        habit["done_dates"].discard(raw)
        return jsonify(_public(habit))

    @app.delete("/habits/<int:habit_id>")
    def delete_habit(habit_id):
        _habit, err = _get_habit(habit_id)
        if err is not None:
            return err
        del app.config["HABITS"][habit_id]
        return jsonify({"deleted": habit_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
