"""Flask events + RSVP API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /events`` — create an event (``{\"title\": str,
  \"date\": \"YYYY-MM-DD\", \"capacity\": int, \"location\": str?}``)
- ``GET /events`` — list events; ``?date=YYYY-MM-DD`` filter
- ``GET /events/<id>`` — fetch one event (includes RSVP counts)
- ``PUT /events/<id>`` — update title, date, capacity and/or location
  (capacity cannot drop below the current number of "yes" RSVPs)
- ``DELETE /events/<id>`` — delete an event and its RSVPs
- ``POST /events/<id>/rsvp`` — RSVP (``{\"name\": str, \"response\":
  \"yes\" | \"no\" | \"maybe\"}``); a "yes" is refused with 409 when the
  event is full; one RSVP per name per event (repeat RSVPs update it)
- ``GET /events/<id>/attendees`` — list RSVPs; ``?response=`` filter
- ``PUT /rsvps/<id>`` — change an RSVP's response (capacity re-checked)
- ``DELETE /rsvps/<id>`` — remove an RSVP
- ``GET /events/<id>/summary`` — counts by response and spots remaining

An event is ``{\"id\": int, \"title\": str, \"date\": str,
\"capacity\": int, \"location\": str | None, \"yes_count\": int,
\"no_count\": int, \"maybe_count\": int, \"spots_remaining\": int}``.
An RSVP is ``{\"id\": int, \"event_id\": int, \"name\": str,
\"response\": \"yes\" | \"no\" | \"maybe\"}``.
"""
from datetime import datetime

from flask import Flask, jsonify, request

RESPONSES = ("yes", "no", "maybe")


def _valid_date(value):
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def create_app():
    """Return a Flask app with a fresh, empty in-memory event store."""
    app = Flask(__name__)
    app.config["EVENTS"] = {}
    app.config["RSVPS"] = {}
    app.config["NEXT_EVENT_ID"] = 1
    app.config["NEXT_RSVP_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _counts(event_id):
        counts = {r: 0 for r in RESPONSES}
        for rsvp in app.config["RSVPS"].values():
            if rsvp["event_id"] == event_id:
                counts[rsvp["response"]] += 1
        return counts

    def _public_event(event):
        counts = _counts(event["id"])
        return {
            "id": event["id"],
            "title": event["title"],
            "date": event["date"],
            "capacity": event["capacity"],
            "location": event["location"],
            "yes_count": counts["yes"],
            "no_count": counts["no"],
            "maybe_count": counts["maybe"],
            "spots_remaining": event["capacity"] - counts["yes"],
        }

    def _get_event(event_id):
        event = app.config["EVENTS"].get(event_id)
        if event is None:
            return None, _error(f"event {event_id} not found", 404)
        return event, None

    def _get_rsvp(rsvp_id):
        rsvp = app.config["RSVPS"].get(rsvp_id)
        if rsvp is None:
            return None, _error(f"rsvp {rsvp_id} not found", 404)
        return rsvp, None

    @app.post("/events")
    def create_event():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        date = data.get("date")
        if not _valid_date(date):
            return _error("date is required and must be YYYY-MM-DD", 400)
        capacity = data.get("capacity")
        if not isinstance(capacity, int) or isinstance(capacity, bool) \
                or capacity <= 0:
            return _error("capacity is required and must be a positive integer", 400)
        location = data.get("location")
        if location is not None and (not isinstance(location, str)
                                     or not location.strip()):
            return _error("location must be a non-empty string", 400)
        event_id = app.config["NEXT_EVENT_ID"]
        app.config["NEXT_EVENT_ID"] += 1
        event = {
            "id": event_id,
            "title": title.strip(),
            "date": date,
            "capacity": capacity,
            "location": location.strip() if location else None,
        }
        app.config["EVENTS"][event_id] = event
        return jsonify(_public_event(event)), 201

    @app.get("/events")
    def list_events():
        events = sorted(app.config["EVENTS"].values(), key=lambda e: e["id"])
        date = request.args.get("date")
        if date is not None:
            if not _valid_date(date):
                return _error("date must be YYYY-MM-DD", 400)
            events = [e for e in events if e["date"] == date]
        return jsonify([_public_event(e) for e in events])

    @app.get("/events/<int:event_id>")
    def get_event(event_id):
        event, err = _get_event(event_id)
        if err is not None:
            return err
        return jsonify(_public_event(event))

    @app.put("/events/<int:event_id>")
    def update_event(event_id):
        event, err = _get_event(event_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("title", "date", "capacity", "location")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide title, date, capacity and/or location",
                400,
            )
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            event["title"] = data["title"].strip()
        if "date" in data:
            if not _valid_date(data["date"]):
                return _error("date must be YYYY-MM-DD", 400)
            event["date"] = data["date"]
        if "location" in data:
            if data["location"] is not None and (
                not isinstance(data["location"], str)
                or not data["location"].strip()
            ):
                return _error("location must be a non-empty string", 400)
            event["location"] = (
                data["location"].strip() if data["location"] else None
            )
        if "capacity" in data:
            cap = data["capacity"]
            if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
                return _error("capacity must be a positive integer", 400)
            if cap < _counts(event_id)["yes"]:
                return _error(
                    "capacity cannot be below the current number of "
                    "'yes' RSVPs",
                    400,
                )
            event["capacity"] = cap
        return jsonify(_public_event(event))

    @app.delete("/events/<int:event_id>")
    def delete_event(event_id):
        _event, err = _get_event(event_id)
        if err is not None:
            return err
        del app.config["EVENTS"][event_id]
        for rsvp_id in [
            r["id"] for r in app.config["RSVPS"].values()
            if r["event_id"] == event_id
        ]:
            del app.config["RSVPS"][rsvp_id]
        return jsonify({"deleted": event_id})

    @app.post("/events/<int:event_id>/rsvp")
    def rsvp(event_id):
        event, err = _get_event(event_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        response = data.get("response")
        if response not in RESPONSES:
            return _error(f"response must be one of {', '.join(RESPONSES)}", 400)
        name = name.strip()
        existing = next(
            (
                r for r in app.config["RSVPS"].values()
                if r["event_id"] == event_id and r["name"].lower() == name.lower()
            ),
            None,
        )
        if response == "yes" and (
            existing is None or existing["response"] != "yes"
        ):
            if _counts(event_id)["yes"] >= event["capacity"]:
                return _error("event is full: no spots remaining", 409)
        if existing is not None:
            existing["response"] = response
            return jsonify(existing)
        rsvp_id = app.config["NEXT_RSVP_ID"]
        app.config["NEXT_RSVP_ID"] += 1
        rsvp = {"id": rsvp_id, "event_id": event_id, "name": name,
                "response": response}
        app.config["RSVPS"][rsvp_id] = rsvp
        return jsonify(rsvp), 201

    @app.get("/events/<int:event_id>/attendees")
    def attendees(event_id):
        _event, err = _get_event(event_id)
        if err is not None:
            return err
        response = request.args.get("response")
        if response is not None and response not in RESPONSES:
            return _error(f"response must be one of {', '.join(RESPONSES)}", 400)
        rows = [r for r in app.config["RSVPS"].values()
                if r["event_id"] == event_id]
        rows.sort(key=lambda r: r["id"])
        if response is not None:
            rows = [r for r in rows if r["response"] == response]
        return jsonify(rows)

    @app.put("/rsvps/<int:rsvp_id>")
    def update_rsvp(rsvp_id):
        rsvp, err = _get_rsvp(rsvp_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "response" not in data:
            return _error("nothing to update: provide response", 400)
        response = data["response"]
        if response not in RESPONSES:
            return _error(f"response must be one of {', '.join(RESPONSES)}", 400)
        if response == "yes" and rsvp["response"] != "yes":
            event = app.config["EVENTS"][rsvp["event_id"]]
            if _counts(event["id"])["yes"] >= event["capacity"]:
                return _error("event is full: no spots remaining", 409)
        rsvp["response"] = response
        return jsonify(rsvp)

    @app.delete("/rsvps/<int:rsvp_id>")
    def delete_rsvp(rsvp_id):
        _rsvp, err = _get_rsvp(rsvp_id)
        if err is not None:
            return err
        del app.config["RSVPS"][rsvp_id]
        return jsonify({"deleted": rsvp_id})

    @app.get("/events/<int:event_id>/summary")
    def summary(event_id):
        event, err = _get_event(event_id)
        if err is not None:
            return err
        counts = _counts(event_id)
        return jsonify(
            {
                "event_id": event_id,
                "title": event["title"],
                "capacity": event["capacity"],
                "yes": counts["yes"],
                "no": counts["no"],
                "maybe": counts["maybe"],
                "total_rsvps": sum(counts.values()),
                "spots_remaining": event["capacity"] - counts["yes"],
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
