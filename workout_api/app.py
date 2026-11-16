"""REST workout tracker API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /workouts`` — list all workouts (``?type=run`` filters by type)
- ``POST /workouts`` — log a workout
- ``GET /workouts/<id>`` — fetch one workout
- ``DELETE /workouts/<id>`` — delete a workout
- ``GET /workouts/weekly`` — totals for one week (``?week=YYYY-MM-DD``
  picks any day inside the week; defaults to the current week)
- ``GET /workouts/summary`` — totals grouped by workout type

A workout is ``{"id": int, "type": str, "duration_minutes": number,
"calories": number, "date": "YYYY-MM-DD"}``. ``date`` defaults to
today when omitted.
"""
from datetime import date, timedelta

from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory workout store."""
    app = Flask(__name__)
    app.config["WORKOUTS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _parse_date(value):
        """Parse a YYYY-MM-DD string, returning (date, None) or (None, error)."""
        if not isinstance(value, str):
            return None, _error("date must be a YYYY-MM-DD string", 400)
        try:
            return date.fromisoformat(value), None
        except ValueError:
            return None, _error("date must be a valid YYYY-MM-DD date", 400)

    def _totals(workouts):
        return {
            "count": len(workouts),
            "total_duration_minutes": round(
                sum(w["duration_minutes"] for w in workouts), 2
            ),
            "total_calories": round(sum(w["calories"] for w in workouts), 2),
        }

    @app.get("/workouts")
    def list_workouts():
        workouts = sorted(app.config["WORKOUTS"].values(), key=lambda w: w["id"])
        workout_type = request.args.get("type")
        if workout_type:
            needle = workout_type.strip().lower()
            workouts = [w for w in workouts if w["type"].lower() == needle]
        return jsonify(workouts)

    @app.post("/workouts")
    def create_workout():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        workout_type = data.get("type")
        if not isinstance(workout_type, str) or not workout_type.strip():
            return _error("type is required and must be a non-empty string", 400)
        duration = data.get("duration_minutes")
        if not _is_number(duration) or duration <= 0:
            return _error("duration_minutes is required and must be a number above 0", 400)
        calories = data.get("calories")
        if not _is_number(calories) or calories < 0:
            return _error("calories is required and must be a number of 0 or more", 400)
        if "date" in data and data["date"] is not None:
            parsed, err = _parse_date(data["date"])
            if err is not None:
                return err
            workout_date = parsed
        else:
            workout_date = date.today()
        workout_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        workout = {
            "id": workout_id,
            "type": workout_type.strip(),
            "duration_minutes": duration,
            "calories": calories,
            "date": workout_date.isoformat(),
        }
        app.config["WORKOUTS"][workout_id] = workout
        return jsonify(workout), 201

    @app.get("/workouts/weekly")
    def weekly_totals():
        week = request.args.get("week")
        if week is None:
            anchor = date.today()
        else:
            anchor, err = _parse_date(week)
            if err is not None:
                return err
        monday = anchor - timedelta(days=anchor.weekday())
        sunday = monday + timedelta(days=6)
        in_week = [
            w
            for w in app.config["WORKOUTS"].values()
            if monday <= date.fromisoformat(w["date"]) <= sunday
        ]
        result = {
            "week_start": monday.isoformat(),
            "week_end": sunday.isoformat(),
            **_totals(in_week),
        }
        return jsonify(result)

    @app.get("/workouts/summary")
    def summary_by_type():
        by_type = {}
        for workout in app.config["WORKOUTS"].values():
            entry = by_type.setdefault(workout["type"], [])
            entry.append(workout)
        return jsonify({t: _totals(ws) for t, ws in sorted(by_type.items())})

    @app.get("/workouts/<int:workout_id>")
    def get_workout(workout_id):
        workout = app.config["WORKOUTS"].get(workout_id)
        if workout is None:
            return _error(f"workout {workout_id} not found", 404)
        return jsonify(workout)

    @app.delete("/workouts/<int:workout_id>")
    def delete_workout(workout_id):
        if workout_id not in app.config["WORKOUTS"]:
            return _error(f"workout {workout_id} not found", 404)
        del app.config["WORKOUTS"][workout_id]
        return jsonify({"deleted": workout_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
