"""Flask volunteer events API: events, signups with capacity limits, and hours logged per volunteer."""
from datetime import datetime

from flask import Flask, jsonify, request


def _parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _bad(message, code=400):
    return jsonify({"error": message}), code


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory volunteer store."""
    app = Flask(__name__)
    events = {}
    signups = {}
    next_event = [1]
    next_signup = [1]

    def _event_view(event):
        mine = [s for s in signups.values() if s["event_id"] == event["id"]]
        return {
            **event,
            "signup_count": len(mine),
            "spots_remaining": max(0, event["capacity"] - len(mine)),
            "total_hours": round(sum(s["hours"] for s in mine), 2),
        }

    @app.route("/events", methods=["POST"])
    def create_event():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        event_date = _parse_date(data.get("date"))
        capacity = data.get("capacity")
        location = data.get("location", "")
        if not isinstance(name, str) or not name.strip():
            return _bad("name is required and must be a non-empty string")
        if event_date is None:
            return _bad("date is required as YYYY-MM-DD")
        if not _is_int(capacity) or capacity <= 0:
            return _bad("capacity is required and must be a positive integer")
        if location is not None and not isinstance(location, str):
            return _bad("location must be a string")
        event_id = next_event[0]
        next_event[0] += 1
        events[event_id] = {
            "id": event_id,
            "name": name.strip(),
            "date": event_date.isoformat(),
            "capacity": capacity,
            "location": location or "",
        }
        return jsonify(_event_view(events[event_id])), 201

    @app.route("/events", methods=["GET"])
    def list_events():
        return jsonify([_event_view(e) for e in events.values()])

    @app.route("/events/<int:event_id>", methods=["GET"])
    def get_event(event_id):
        event = events.get(event_id)
        if event is None:
            return _bad("event not found", 404)
        return jsonify(_event_view(event))

    @app.route("/events/<int:event_id>", methods=["DELETE"])
    def delete_event(event_id):
        if event_id not in events:
            return _bad("event not found", 404)
        del events[event_id]
        for sid in [k for k, v in signups.items() if v["event_id"] == event_id]:
            del signups[sid]
        return jsonify({"deleted": event_id})

    @app.route("/events/<int:event_id>/signups", methods=["POST"])
    def create_signup(event_id):
        event = events.get(event_id)
        if event is None:
            return _bad("event not found", 404)
        data = request.get_json(silent=True) or {}
        name = data.get("volunteer_name")
        email = data.get("email", "")
        if not isinstance(name, str) or not name.strip():
            return _bad("volunteer_name is required and must be a non-empty string")
        if email is not None and not isinstance(email, str):
            return _bad("email must be a string")
        mine = [s for s in signups.values() if s["event_id"] == event_id]
        if any(s["volunteer_name"].strip().lower() == name.strip().lower() for s in mine):
            return _bad("volunteer is already signed up for this event", 409)
        if len(mine) >= event["capacity"]:
            return _bad("event is full", 409)
        signup_id = next_signup[0]
        next_signup[0] += 1
        signups[signup_id] = {
            "id": signup_id,
            "event_id": event_id,
            "volunteer_name": name.strip(),
            "email": email or "",
            "hours": 0,
        }
        return jsonify(signups[signup_id]), 201

    @app.route("/events/<int:event_id>/signups", methods=["GET"])
    def list_signups(event_id):
        if event_id not in events:
            return _bad("event not found", 404)
        return jsonify([s for s in signups.values() if s["event_id"] == event_id])

    @app.route("/events/<int:event_id>/signups/<int:signup_id>/hours", methods=["POST"])
    def log_hours(event_id, signup_id):
        signup = signups.get(signup_id)
        if signup is None or signup["event_id"] != event_id:
            return _bad("signup not found", 404)
        data = request.get_json(silent=True) or {}
        hours = data.get("hours")
        if not _is_number(hours) or hours <= 0:
            return _bad("hours is required and must be a positive number")
        signup["hours"] = round(signup["hours"] + hours, 2)
        return jsonify(signup)

    @app.route("/volunteers", methods=["GET"])
    def volunteer_hours():
        name = request.args.get("name")
        if not name:
            return _bad("name query parameter is required")
        mine = [s for s in signups.values() if s["volunteer_name"].lower() == name.lower()]
        event_names = []
        for signup in mine:
            event = events.get(signup["event_id"])
            if event is not None:
                event_names.append(event["name"])
        return jsonify({
            "volunteer_name": name,
            "signup_count": len(mine),
            "total_hours": round(sum(s["hours"] for s in mine), 2),
            "events": event_names,
        })

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "route not found"}), 404

    return app
