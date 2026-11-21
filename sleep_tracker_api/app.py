"""Flask sleep log API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /nights`` — log a night (``{"date": "YYYY-MM-DD",
  "bedtime": "HH:MM", "wake_time": "HH:MM", "quality": 1-5,
  "notes": str?}``); duration is computed from bedtime/wake time and
  may cross midnight. One entry per date (a second entry for the same
  date is refused with 409).
- ``GET /nights`` — list nights, newest first; ``?date=YYYY-MM-DD``
  and ``?min_quality=1-5`` filters
- ``GET /nights/<id>`` — fetch one night
- ``PUT /nights/<id>`` — update date, bedtime, wake_time, quality
  and/or notes (duration is recomputed)
- ``DELETE /nights/<id>`` — delete a night
- ``GET /stats`` — totals, average duration/quality, longest and
  shortest nights, and the current streak of consecutive logged days

A night is ``{"id": int, "date": str, "bedtime": str, "wake_time": str,
"duration_minutes": int, "duration_hours": float, "quality": int,
"notes": str | None}``.
"""
from datetime import datetime, timedelta

from flask import Flask, jsonify, request


def _valid_date(value):
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _valid_time(value):
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%H:%M")
    except ValueError:
        return False
    return True


def _duration_minutes(bedtime, wake_time):
    """Minutes slept, allowing wake time to fall after midnight."""
    bed = datetime.strptime(bedtime, "%H:%M")
    wake = datetime.strptime(wake_time, "%H:%M")
    minutes = int((wake - bed).total_seconds() // 60) % 1440
    return minutes


def create_app():
    """Return a Flask app with a fresh, empty in-memory night store."""
    app = Flask(__name__)
    app.config["NIGHTS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _public(night):
        return {
            "id": night["id"],
            "date": night["date"],
            "bedtime": night["bedtime"],
            "wake_time": night["wake_time"],
            "duration_minutes": night["duration_minutes"],
            "duration_hours": round(night["duration_minutes"] / 60, 2),
            "quality": night["quality"],
            "notes": night["notes"],
        }

    def _get_night(night_id):
        night = app.config["NIGHTS"].get(night_id)
        if night is None:
            return None, _error(f"night {night_id} not found", 404)
        return night, None

    def _validate(data, partial=False):
        """Validate night fields; returns (cleaned, error_response)."""
        cleaned = {}
        if not partial or "date" in data:
            date = data.get("date")
            if not _valid_date(date):
                return None, _error("date is required and must be YYYY-MM-DD", 400)
            cleaned["date"] = date
        if not partial or "bedtime" in data:
            bedtime = data.get("bedtime")
            if not _valid_time(bedtime):
                return None, _error("bedtime is required and must be HH:MM", 400)
            cleaned["bedtime"] = bedtime
        if not partial or "wake_time" in data:
            wake_time = data.get("wake_time")
            if not _valid_time(wake_time):
                return None, _error("wake_time is required and must be HH:MM", 400)
            cleaned["wake_time"] = wake_time
        if not partial or "quality" in data:
            quality = data.get("quality")
            if not isinstance(quality, int) or isinstance(quality, bool) \
                    or quality < 1 or quality > 5:
                return None, _error("quality is required and must be an integer 1-5", 400)
            cleaned["quality"] = quality
        if "notes" in data:
            notes = data["notes"]
            if notes is not None and (not isinstance(notes, str)
                                      or not notes.strip()):
                return None, _error("notes must be a non-empty string", 400)
            cleaned["notes"] = notes.strip() if notes else None
        return cleaned, None

    def _check_duration(bedtime, wake_time):
        minutes = _duration_minutes(bedtime, wake_time)
        if minutes == 0:
            return None, _error(
                "bedtime and wake_time must differ: duration would be 0 minutes",
                400,
            )
        return minutes, None

    def _duplicate_date(date, exclude_id=None):
        return any(
            n["date"] == date and n["id"] != exclude_id
            for n in app.config["NIGHTS"].values()
        )

    @app.post("/nights")
    def create_night():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        cleaned, err = _validate(data)
        if err is not None:
            return err
        minutes, err = _check_duration(cleaned["bedtime"], cleaned["wake_time"])
        if err is not None:
            return err
        if _duplicate_date(cleaned["date"]):
            return _error(
                f"a night is already logged for {cleaned['date']}", 409
            )
        night_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        night = {
            "id": night_id,
            "date": cleaned["date"],
            "bedtime": cleaned["bedtime"],
            "wake_time": cleaned["wake_time"],
            "duration_minutes": minutes,
            "quality": cleaned["quality"],
            "notes": cleaned.get("notes"),
        }
        app.config["NIGHTS"][night_id] = night
        return jsonify(_public(night)), 201

    @app.get("/nights")
    def list_nights():
        nights = sorted(
            app.config["NIGHTS"].values(), key=lambda n: n["date"], reverse=True
        )
        date = request.args.get("date")
        if date is not None:
            if not _valid_date(date):
                return _error("date must be YYYY-MM-DD", 400)
            nights = [n for n in nights if n["date"] == date]
        min_quality = request.args.get("min_quality")
        if min_quality is not None:
            try:
                min_quality = int(min_quality)
            except ValueError:
                return _error("min_quality must be an integer 1-5", 400)
            if min_quality < 1 or min_quality > 5:
                return _error("min_quality must be an integer 1-5", 400)
            nights = [n for n in nights if n["quality"] >= min_quality]
        return jsonify([_public(n) for n in nights])

    @app.get("/nights/<int:night_id>")
    def get_night(night_id):
        night, err = _get_night(night_id)
        if err is not None:
            return err
        return jsonify(_public(night))

    @app.put("/nights/<int:night_id>")
    def update_night(night_id):
        night, err = _get_night(night_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("date", "bedtime", "wake_time", "quality", "notes")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide date, bedtime, wake_time, "
                "quality and/or notes",
                400,
            )
        cleaned, err = _validate(data, partial=True)
        if err is not None:
            return err
        new_date = cleaned.get("date", night["date"])
        new_bed = cleaned.get("bedtime", night["bedtime"])
        new_wake = cleaned.get("wake_time", night["wake_time"])
        minutes, err = _check_duration(new_bed, new_wake)
        if err is not None:
            return err
        if _duplicate_date(new_date, exclude_id=night_id):
            return _error(f"a night is already logged for {new_date}", 409)
        night.update(cleaned)
        night["duration_minutes"] = minutes
        return jsonify(_public(night))

    @app.delete("/nights/<int:night_id>")
    def delete_night(night_id):
        _night, err = _get_night(night_id)
        if err is not None:
            return err
        del app.config["NIGHTS"][night_id]
        return jsonify({"deleted": night_id})

    @app.get("/stats")
    def stats():
        nights = list(app.config["NIGHTS"].values())
        if not nights:
            return jsonify(
                {
                    "total_nights": 0,
                    "total_sleep_minutes": 0,
                    "average_duration_minutes": None,
                    "average_duration_hours": None,
                    "average_quality": None,
                    "longest_night_minutes": None,
                    "shortest_night_minutes": None,
                    "streak_days": 0,
                }
            )
        durations = [n["duration_minutes"] for n in nights]
        total = sum(durations)
        avg_minutes = round(total / len(nights), 1)
        avg_quality = round(
            sum(n["quality"] for n in nights) / len(nights), 2
        )
        dates = sorted(
            {datetime.strptime(n["date"], "%Y-%m-%d").date() for n in nights},
            reverse=True,
        )
        streak = 1
        for i in range(1, len(dates)):
            if dates[i] == dates[i - 1] - timedelta(days=1):
                streak += 1
            else:
                break
        return jsonify(
            {
                "total_nights": len(nights),
                "total_sleep_minutes": total,
                "average_duration_minutes": avg_minutes,
                "average_duration_hours": round(avg_minutes / 60, 2),
                "average_quality": avg_quality,
                "longest_night_minutes": max(durations),
                "shortest_night_minutes": min(durations),
                "streak_days": streak,
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
