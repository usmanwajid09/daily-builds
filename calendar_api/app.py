"""Calendar events API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /events`` — create an event
  (``{"title": str, "start": "YYYY-MM-DDTHH:MM", "end": "YYYY-MM-DDTHH:MM",
  "location": str (optional), "description": str (optional)}``)
- ``GET /events`` — list all events, sorted by start
  (``?date=YYYY-MM-DD`` filters to events on that day)
- ``GET /events/upcoming`` — events starting at or after ``now``
  (``?now=YYYY-MM-DDTHH:MM``; defaults to the current time)
- ``GET /events/<id>`` — fetch one event
- ``PUT /events/<id>`` — replace an event
- ``DELETE /events/<id>`` — delete an event

An event whose end is not after its start returns 400, as do bad
timestamps. Unknown event ids return 404.
"""
from datetime import datetime

from flask import Flask, jsonify, request

_FMT = "%Y-%m-%dT%H:%M"


def create_app():
    """Return a Flask app with a fresh, empty in-memory event store."""
    app = Flask(__name__)
    app.config["EVENTS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_time(value, field):
        if not isinstance(value, str):
            return None, f"{field} must be a string like '2026-11-17T09:00'"
        try:
            return datetime.strptime(value, _FMT), None
        except ValueError:
            return None, (
                f"{field} must be a valid datetime like '2026-11-17T09:00'"
            )

    def _validate(data):
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return None, "title is required and must be a non-empty string"
        start, err = _parse_time(data.get("start"), "start")
        if err:
            return None, err
        end, err = _parse_time(data.get("end"), "end")
        if err:
            return None, err
        if end <= start:
            return None, "end must be after start"
        location = data.get("location")
        if location is not None and not isinstance(location, str):
            return None, "location must be a string"
        description = data.get("description")
        if description is not None and not isinstance(description, str):
            return None, "description must be a string"
        event = {
            "title": title.strip(),
            "start": start.strftime(_FMT),
            "end": end.strftime(_FMT),
            "location": location,
            "description": description,
        }
        return event, None

    def _get_event(event_id):
        event = app.config["EVENTS"].get(event_id)
        if event is None:
            return None, _error(f"event {event_id} not found", 404)
        return event, None

    @app.post("/events")
    def create_event():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        event, err = _validate(data)
        if err:
            return _error(err, 400)
        event_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        event["id"] = event_id
        app.config["EVENTS"][event_id] = event
        return jsonify(event), 201

    @app.get("/events")
    def list_events():
        events = sorted(
            app.config["EVENTS"].values(), key=lambda e: (e["start"], e["id"])
        )
        date = request.args.get("date")
        if date:
            try:
                day = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                return _error("date must be like '2026-11-17'", 400)
            events = [
                e
                for e in events
                if datetime.strptime(e["start"], _FMT).date() == day
            ]
        return jsonify(events)

    @app.get("/events/upcoming")
    def upcoming_events():
        now_raw = request.args.get("now")
        if now_raw:
            now, err = _parse_time(now_raw, "now")
            if err:
                return _error(err, 400)
        else:
            now = datetime.now()
        events = [
            e
            for e in app.config["EVENTS"].values()
            if datetime.strptime(e["start"], _FMT) >= now
        ]
        events.sort(key=lambda e: (e["start"], e["id"]))
        return jsonify(events)

    @app.get("/events/<int:event_id>")
    def get_event(event_id):
        event, err = _get_event(event_id)
        if err is not None:
            return err
        return jsonify(event)

    @app.put("/events/<int:event_id>")
    def update_event(event_id):
        event, err = _get_event(event_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        updated, err = _validate(data)
        if err:
            return _error(err, 400)
        updated["id"] = event_id
        app.config["EVENTS"][event_id] = updated
        return jsonify(updated)

    @app.delete("/events/<int:event_id>")
    def delete_event(event_id):
        event, err = _get_event(event_id)
        if err is not None:
            return err
        del app.config["EVENTS"][event_id]
        return jsonify({"deleted": event_id})

    return app
