"""Flask parking API.

In-memory store, fresh for each app instance. All responses are JSON.

Spots — ``POST /spots`` (``{"spot_number": str, "zone": str?,
"vehicle_type": str?, "hourly_rate": number?}``); ``GET /spots`` with
``?zone=``, ``?vehicle_type=`` and ``?status=`` filters;
``GET /spots/<id>``; ``PUT /spots/<id>``;
``DELETE /spots/<id>`` (only while free and with no sessions).

Vehicle types: ``car``, ``motorcycle``, ``truck``. A spot's ``status`` is
``free`` or ``occupied``.

Sessions — ``POST /spots/<id>/check-in`` (``{"license_plate": str,
"vehicle_type": str?}``); ``POST /spots/<id>/check-out``
(``{"ended_at": str?}`` — ISO timestamp, defaults to now);
``GET /spots/<id>/sessions``; ``GET /sessions``;
``GET /sessions/<id>``.

Fees: parking is charged per started hour (``ceil(duration_hours)``),
with a minimum of one hour, at the spot's ``hourly_rate``. A check-out
returns the session with ``duration_minutes``, ``hours_billed`` and
``fee``. ``GET /summary`` reports ``total_spots``, ``occupied_spots``,
``free_spots``, ``occupancy_percent`` and ``total_revenue`` (fees from
completed sessions only).
"""
import math
from datetime import datetime, timezone

from flask import Flask, jsonify, request

VEHICLE_TYPES = ("car", "motorcycle", "truck")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value, field):
    if not isinstance(value, str) or not value.strip():
        return None, f"{field} must be an ISO 8601 timestamp string"
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None, f"{field} must be an ISO 8601 timestamp string"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed, None


def create_app():
    """Return a Flask app with a fresh, empty in-memory parking store."""
    app = Flask(__name__)
    app.config["SPOTS"] = {}
    app.config["SESSIONS"] = {}
    app.config["NEXT_SPOT_ID"] = 1
    app.config["NEXT_SESSION_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_spot(spot_id):
        spot = app.config["SPOTS"].get(spot_id)
        if spot is None:
            return None, _error(f"spot {spot_id} not found", 404)
        return spot, None

    def _active_session(spot_id):
        for session in app.config["SESSIONS"].values():
            if session["spot_id"] == spot_id and session["status"] == "active":
                return session
        return None

    def _summary():
        spots = list(app.config["SPOTS"].values())
        occupied = sum(1 for s in spots if s["status"] == "occupied")
        revenue = round(
            sum(
                s["fee"]
                for s in app.config["SESSIONS"].values()
                if s["status"] == "completed" and s["fee"] is not None
            ),
            2,
        )
        return {
            "total_spots": len(spots),
            "occupied_spots": occupied,
            "free_spots": len(spots) - occupied,
            "occupancy_percent": round(
                occupied / len(spots) * 100, 2
            ) if spots else 0.0,
            "total_revenue": revenue,
        }

    @app.post("/spots")
    def create_spot():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        spot_number = data.get("spot_number")
        if not isinstance(spot_number, str) or not spot_number.strip():
            return _error(
                "spot_number is required and must be a non-empty string", 400
            )
        if any(
            s["spot_number"].lower() == spot_number.strip().lower()
            for s in app.config["SPOTS"].values()
        ):
            return _error(f"spot_number {spot_number!r} already exists", 409)
        zone = data.get("zone")
        if zone is not None and (not isinstance(zone, str) or not zone.strip()):
            return _error("zone must be a non-empty string", 400)
        vehicle_type = data.get("vehicle_type", "car")
        if vehicle_type not in VEHICLE_TYPES:
            return _error(
                f"vehicle_type must be one of {', '.join(VEHICLE_TYPES)}", 400
            )
        hourly_rate = data.get("hourly_rate", 100)
        if (
            isinstance(hourly_rate, bool)
            or not isinstance(hourly_rate, (int, float))
            or hourly_rate < 0
        ):
            return _error("hourly_rate must be a non-negative number", 400)
        spot_id = app.config["NEXT_SPOT_ID"]
        app.config["NEXT_SPOT_ID"] += 1
        spot = {
            "id": spot_id,
            "spot_number": spot_number.strip(),
            "zone": zone.strip() if zone else None,
            "vehicle_type": vehicle_type,
            "hourly_rate": round(float(hourly_rate), 2),
            "status": "free",
        }
        app.config["SPOTS"][spot_id] = spot
        return jsonify(spot), 201

    @app.get("/spots")
    def list_spots():
        spots = list(app.config["SPOTS"].values())
        zone = request.args.get("zone")
        if zone is not None:
            spots = [s for s in spots if (s["zone"] or "").lower() == zone.lower()]
        vehicle_type = request.args.get("vehicle_type")
        if vehicle_type is not None:
            if vehicle_type not in VEHICLE_TYPES:
                return _error(
                    f"vehicle_type filter must be one of "
                    f"{', '.join(VEHICLE_TYPES)}",
                    400,
                )
            spots = [s for s in spots if s["vehicle_type"] == vehicle_type]
        status = request.args.get("status")
        if status is not None:
            if status not in ("free", "occupied"):
                return _error("status filter must be free or occupied", 400)
            spots = [s for s in spots if s["status"] == status]
        return jsonify(spots)

    @app.get("/spots/<int:spot_id>")
    def get_spot(spot_id):
        spot, err = _get_spot(spot_id)
        if err is not None:
            return err
        return jsonify(spot)

    @app.put("/spots/<int:spot_id>")
    def update_spot(spot_id):
        spot, err = _get_spot(spot_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "spot_number" in data:
            new_number = data["spot_number"]
            if not isinstance(new_number, str) or not new_number.strip():
                return _error("spot_number must be a non-empty string", 400)
            if any(
                s["id"] != spot_id
                and s["spot_number"].lower() == new_number.strip().lower()
                for s in app.config["SPOTS"].values()
            ):
                return _error(f"spot_number {new_number!r} already exists", 409)
            spot["spot_number"] = new_number.strip()
        if "zone" in data:
            zone = data["zone"]
            if zone is not None and (not isinstance(zone, str) or not zone.strip()):
                return _error("zone must be a non-empty string", 400)
            spot["zone"] = zone.strip() if zone else None
        if "vehicle_type" in data:
            if data["vehicle_type"] not in VEHICLE_TYPES:
                return _error(
                    f"vehicle_type must be one of {', '.join(VEHICLE_TYPES)}", 400
                )
            spot["vehicle_type"] = data["vehicle_type"]
        if "hourly_rate" in data:
            rate = data["hourly_rate"]
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or rate < 0
            ):
                return _error("hourly_rate must be a non-negative number", 400)
            spot["hourly_rate"] = round(float(rate), 2)
        return jsonify(spot)

    @app.delete("/spots/<int:spot_id>")
    def delete_spot(spot_id):
        spot, err = _get_spot(spot_id)
        if err is not None:
            return err
        if spot["status"] == "occupied":
            return _error("cannot delete an occupied spot", 409)
        if any(
            s["spot_id"] == spot_id for s in app.config["SESSIONS"].values()
        ):
            return _error("cannot delete a spot that has parking sessions", 409)
        del app.config["SPOTS"][spot_id]
        return jsonify({"deleted": spot_id})

    @app.post("/spots/<int:spot_id>/check-in")
    def check_in(spot_id):
        spot, err = _get_spot(spot_id)
        if err is not None:
            return err
        if spot["status"] == "occupied":
            return _error("spot is already occupied", 409)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        license_plate = data.get("license_plate")
        if not isinstance(license_plate, str) or not license_plate.strip():
            return _error(
                "license_plate is required and must be a non-empty string", 400
            )
        vehicle_type = data.get("vehicle_type", spot["vehicle_type"])
        if vehicle_type not in VEHICLE_TYPES:
            return _error(
                f"vehicle_type must be one of {', '.join(VEHICLE_TYPES)}", 400
            )
        if vehicle_type != spot["vehicle_type"]:
            return _error(
                f"spot is for {spot['vehicle_type']} vehicles, "
                f"not {vehicle_type}",
                409,
            )
        started_at = data.get("started_at")
        if started_at is not None:
            start, parse_err = _parse_time(started_at, "started_at")
            if parse_err is not None:
                return _error(parse_err, 400)
            started_iso = start.isoformat()
        else:
            started_iso = _now_iso()
        session_id = app.config["NEXT_SESSION_ID"]
        app.config["NEXT_SESSION_ID"] += 1
        session = {
            "id": session_id,
            "spot_id": spot_id,
            "license_plate": license_plate.strip(),
            "vehicle_type": vehicle_type,
            "started_at": started_iso,
            "ended_at": None,
            "duration_minutes": None,
            "hours_billed": None,
            "fee": None,
            "status": "active",
        }
        app.config["SESSIONS"][session_id] = session
        spot["status"] = "occupied"
        return jsonify(session), 201

    @app.post("/spots/<int:spot_id>/check-out")
    def check_out(spot_id):
        spot, err = _get_spot(spot_id)
        if err is not None:
            return err
        session = _active_session(spot_id)
        if session is None:
            return _error("spot has no active parking session", 409)
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        ended_at = data.get("ended_at")
        if ended_at is not None:
            end, parse_err = _parse_time(ended_at, "ended_at")
            if parse_err is not None:
                return _error(parse_err, 400)
            end_iso = end.isoformat()
        else:
            end = datetime.now(timezone.utc)
            end_iso = end.isoformat()
        start = datetime.fromisoformat(session["started_at"])
        if end < start:
            return _error("ended_at must not be before started_at", 400)
        duration_minutes = (end - start).total_seconds() / 60
        hours_billed = max(1, math.ceil(duration_minutes / 60))
        session["ended_at"] = end_iso
        session["duration_minutes"] = round(duration_minutes, 2)
        session["hours_billed"] = hours_billed
        session["fee"] = round(hours_billed * spot["hourly_rate"], 2)
        session["status"] = "completed"
        spot["status"] = "free"
        return jsonify(session)

    @app.get("/spots/<int:spot_id>/sessions")
    def list_spot_sessions(spot_id):
        spot, err = _get_spot(spot_id)
        if err is not None:
            return err
        return jsonify(
            [s for s in app.config["SESSIONS"].values() if s["spot_id"] == spot_id]
        )

    @app.get("/sessions")
    def list_sessions():
        sessions = list(app.config["SESSIONS"].values())
        status = request.args.get("status")
        if status is not None:
            if status not in ("active", "completed"):
                return _error("status filter must be active or completed", 400)
            sessions = [s for s in sessions if s["status"] == status]
        return jsonify(sessions)

    @app.get("/sessions/<int:session_id>")
    def get_session(session_id):
        session = app.config["SESSIONS"].get(session_id)
        if session is None:
            return _error(f"session {session_id} not found", 404)
        return jsonify(session)

    @app.get("/summary")
    def summary():
        return jsonify(_summary())

    return app
