"""REST daily-water-intake API built with Flask.

In-memory store, fresh for each app instance. Log water in millilitres
(or glasses, 1 glass = 250 ml), compare each day against a daily target
(default 2000 ml), and track the streak of consecutive days meeting the
target.

Endpoints:
- ``GET /target`` — current daily target in ml
- ``PUT /target`` — set the daily target (``{"target_ml": n}``)
- ``POST /intake`` — log water (``{"amount_ml": n}`` or ``{"glasses": n}``,
  optional ``"date": "YYYY-MM-DD"``, defaults to today)
- ``GET /intake?date=YYYY-MM-DD`` — entries and total for a day
- ``GET /summary/<date>`` — total vs target for one day
- ``GET /streak`` — consecutive days meeting the target
- ``GET /weekly?date=YYYY-MM-DD`` — 7-day summary ending on ``date``
"""
import re
from datetime import date, timedelta

from flask import Flask, jsonify, request

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
GLASS_ML = 250


def create_app():
    """Return a Flask app with a fresh, empty in-memory water store."""
    app = Flask(__name__)
    app.config["TARGET_ML"] = 2000
    app.config["INTAKE"] = {}  # date string -> list of {"amount_ml": n}
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

    def _total(day):
        return sum(e["amount_ml"] for e in app.config["INTAKE"].get(day, []))

    def _day_summary(day):
        total = _total(day)
        target = app.config["TARGET_ML"]
        return {
            "date": day,
            "total_ml": total,
            "target_ml": target,
            "met_target": total >= target,
            "remaining_ml": max(0, target - total),
        }

    @app.get("/target")
    def get_target():
        return jsonify({"target_ml": app.config["TARGET_ML"]})

    @app.put("/target")
    def set_target():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        target = data.get("target_ml")
        if not isinstance(target, (int, float)) or isinstance(target, bool) or target <= 0:
            return _error("target_ml is required and must be a positive number", 400)
        app.config["TARGET_ML"] = target
        return jsonify({"target_ml": target})

    @app.post("/intake")
    def log_intake():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        day = data.get("date", date.today().isoformat())
        if not _valid_date(day):
            return _error("date must be a valid YYYY-MM-DD date", 400)
        amount = data.get("amount_ml")
        if amount is None and data.get("glasses") is not None:
            glasses = data.get("glasses")
            if not isinstance(glasses, (int, float)) or isinstance(glasses, bool) or glasses <= 0:
                return _error("glasses must be a positive number", 400)
            amount = glasses * GLASS_ML
        if not isinstance(amount, (int, float)) or isinstance(amount, bool) or amount <= 0:
            return _error("amount_ml is required and must be a positive number (or give glasses)", 400)
        entry_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        entry = {"id": entry_id, "date": day, "amount_ml": amount}
        app.config["INTAKE"].setdefault(day, []).append(entry)
        return jsonify({**entry, "day_total_ml": _total(day)}), 201

    @app.get("/intake")
    def list_intake():
        day = request.args.get("date", date.today().isoformat())
        if not _valid_date(day):
            return _error("date must be a valid YYYY-MM-DD date", 400)
        entries = app.config["INTAKE"].get(day, [])
        return jsonify({"date": day, "entries": entries, "total_ml": _total(day)})

    @app.get("/summary/<day>")
    def day_summary(day):
        if not _valid_date(day):
            return _error("date must be a valid YYYY-MM-DD date", 400)
        return jsonify(_day_summary(day))

    @app.get("/streak")
    def streak():
        target = app.config["TARGET_ML"]
        logged = sorted(app.config["INTAKE"])
        if not logged:
            return jsonify({"streak_days": 0, "target_ml": target})
        # Streak ends on the latest logged day; count back while each day met target.
        current = date.fromisoformat(logged[-1])
        count = 0
        while _total(current.isoformat()) >= target:
            count += 1
            current -= timedelta(days=1)
        return jsonify({"streak_days": count, "target_ml": target})

    @app.get("/weekly")
    def weekly():
        end = request.args.get("date", date.today().isoformat())
        if not _valid_date(end):
            return _error("date must be a valid YYYY-MM-DD date", 400)
        end_date = date.fromisoformat(end)
        days = [
            _day_summary((end_date - timedelta(days=i)).isoformat())
            for i in range(6, -1, -1)
        ]
        met = sum(1 for d in days if d["met_target"])
        return jsonify(
            {
                "end_date": end,
                "days": days,
                "days_met_target": met,
                "total_ml": sum(d["total_ml"] for d in days),
                "average_ml_per_day": round(sum(d["total_ml"] for d in days) / 7, 1),
            }
        )

    return app
