"""REST mood tracker API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. One entry per date.

An entry is ``{"date": "YYYY-MM-DD", "mood": int 1-5, "note": str,
"tags": [str, ...]}``. ``note`` and ``tags`` are optional on create
(default ``""`` and ``[]``).

Endpoints:

- ``POST /entries`` — create an entry (201; duplicate date -> 409)
- ``GET /entries`` — list all entries, sorted by date ascending
- ``GET /entries/<date>`` — fetch one entry (unknown date -> 404)
- ``PUT /entries/<date>`` — update mood/note/tags (unknown -> 404)
- ``DELETE /entries/<date>`` — delete an entry (unknown -> 404)
- ``GET /stats/weekly`` — stats for the last 7 days
- ``GET /stats/trend`` — daily moods and direction over the last
  ``?days=`` days (default 14)
- ``GET /stats/tags`` — tag counts, most frequent first
"""
import re
from collections import Counter
from datetime import date, timedelta

from flask import Flask, jsonify, request

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def create_app():
    """Return a Flask app with a fresh, empty in-memory mood store."""
    app = Flask(__name__)
    app.config["ENTRIES"] = {}

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_date(value):
        """Parse a YYYY-MM-DD string, returning (date, None) or (None, error)."""
        if not isinstance(value, str) or not _DATE_RE.match(value):
            return None, _error("date must be a YYYY-MM-DD string", 400)
        try:
            return date.fromisoformat(value), None
        except ValueError:
            return None, _error("date must be a valid YYYY-MM-DD date", 400)

    def _validate_mood(value):
        if isinstance(value, bool) or not isinstance(value, int):
            return _error("mood must be an integer between 1 and 5", 400)
        if value < 1 or value > 5:
            return _error("mood must be between 1 and 5", 400)
        return None

    def _validate_note(value):
        if not isinstance(value, str):
            return _error("note must be a string", 400)
        return None

    def _validate_tags(value):
        if not isinstance(value, list) or any(
            not isinstance(tag, str) for tag in value
        ):
            return _error("tags must be a list of strings", 400)
        return None

    def _entry_dict(entry):
        return {
            "date": entry["date"],
            "mood": entry["mood"],
            "note": entry["note"],
            "tags": list(entry["tags"]),
        }

    def _average(moods):
        if not moods:
            return None
        return round(sum(moods) / len(moods), 2)

    @app.post("/entries")
    def create_entry():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        parsed, err = _parse_date(data.get("date"))
        if err is not None:
            return err
        if "mood" not in data:
            return _error("mood is required", 400)
        err = _validate_mood(data["mood"])
        if err is not None:
            return err
        note = data.get("note", "")
        err = _validate_note(note)
        if err is not None:
            return err
        tags = data.get("tags", [])
        err = _validate_tags(tags)
        if err is not None:
            return err
        date_str = parsed.isoformat()
        if date_str in app.config["ENTRIES"]:
            return _error(f"an entry already exists for {date_str}", 409)
        entry = {"date": date_str, "mood": data["mood"], "note": note, "tags": tags}
        app.config["ENTRIES"][date_str] = entry
        return jsonify(_entry_dict(entry)), 201

    @app.get("/entries")
    def list_entries():
        entries = sorted(app.config["ENTRIES"].values(), key=lambda e: e["date"])
        return jsonify([_entry_dict(e) for e in entries])

    @app.get("/entries/<date_str>")
    def get_entry(date_str):
        parsed, err = _parse_date(date_str)
        if err is not None:
            return err
        entry = app.config["ENTRIES"].get(parsed.isoformat())
        if entry is None:
            return _error(f"no entry for {parsed.isoformat()}", 404)
        return jsonify(_entry_dict(entry))

    @app.put("/entries/<date_str>")
    def update_entry(date_str):
        parsed, err = _parse_date(date_str)
        if err is not None:
            return err
        entry = app.config["ENTRIES"].get(parsed.isoformat())
        if entry is None:
            return _error(f"no entry for {parsed.isoformat()}", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "mood" in data:
            err = _validate_mood(data["mood"])
            if err is not None:
                return err
        if "note" in data:
            err = _validate_note(data["note"])
            if err is not None:
                return err
        if "tags" in data:
            err = _validate_tags(data["tags"])
            if err is not None:
                return err
        if "mood" in data:
            entry["mood"] = data["mood"]
        if "note" in data:
            entry["note"] = data["note"]
        if "tags" in data:
            entry["tags"] = data["tags"]
        return jsonify(_entry_dict(entry))

    @app.delete("/entries/<date_str>")
    def delete_entry(date_str):
        parsed, err = _parse_date(date_str)
        if err is not None:
            return err
        entry = app.config["ENTRIES"].pop(parsed.isoformat(), None)
        if entry is None:
            return _error(f"no entry for {parsed.isoformat()}", 404)
        return jsonify(_entry_dict(entry))

    @app.get("/stats/weekly")
    def weekly_stats():
        today = date.today()
        start = today - timedelta(days=6)
        moods = [
            e["mood"]
            for e in app.config["ENTRIES"].values()
            if start <= date.fromisoformat(e["date"]) <= today
        ]
        return jsonify(
            {
                "days": 7,
                "entries": len(moods),
                "average_mood": _average(moods),
                "min_mood": min(moods) if moods else None,
                "max_mood": max(moods) if moods else None,
            }
        )

    @app.get("/stats/trend")
    def trend_stats():
        raw_days = request.args.get("days", "14")
        try:
            days = int(raw_days)
        except (TypeError, ValueError):
            return _error("days must be a positive integer", 400)
        if str(raw_days).strip() != str(days) or days < 1:
            return _error("days must be a positive integer", 400)
        today = date.today()
        start = today - timedelta(days=days - 1)
        window = [
            e
            for e in app.config["ENTRIES"].values()
            if start <= date.fromisoformat(e["date"]) <= today
        ]
        window.sort(key=lambda e: e["date"])
        daily = [{"date": e["date"], "mood": e["mood"]} for e in window]
        half = days // 2
        first_dates = {start + timedelta(days=i) for i in range(half)}
        second_dates = {start + timedelta(days=i) for i in range(half, days)}
        first_moods = [
            e["mood"]
            for e in window
            if date.fromisoformat(e["date"]) in first_dates
        ]
        second_moods = [
            e["mood"]
            for e in window
            if date.fromisoformat(e["date"]) in second_dates
        ]
        first_avg = _average(first_moods)
        second_avg = _average(second_moods)
        if first_avg is None or second_avg is None:
            direction = "stable"
        elif second_avg - first_avg >= 0.5:
            direction = "improving"
        elif second_avg - first_avg <= -0.5:
            direction = "declining"
        else:
            direction = "stable"
        return jsonify(
            {
                "days": days,
                "daily": daily,
                "first_half_average": first_avg,
                "second_half_average": second_avg,
                "direction": direction,
            }
        )

    @app.get("/stats/tags")
    def tag_stats():
        counts = Counter()
        for entry in app.config["ENTRIES"].values():
            counts.update(entry["tags"])
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return jsonify([{"tag": tag, "count": count} for tag, count in ranked])

    return app
