"""REST hotel-booking API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /rooms`` — add a room (``{"number": str, "type": str,
  "nightly_rate": number > 0, "capacity": int > 0}``; room numbers
  must be unique)
- ``GET /rooms`` — list all rooms
- ``GET /rooms/<id>`` — fetch one room
- ``GET /rooms/<id>/availability`` — is the room free for a date
  range (``?check_in=YYYY-MM-DD&check_out=YYYY-MM-DD``)
- ``GET /availability`` — rooms free for a date range
  (``?check_in=...&check_out=...``)
- ``POST /bookings`` — book a room (``{"room_id": int,
  "guest_name": str, "check_in": "YYYY-MM-DD",
  "check_out": "YYYY-MM-DD", "guests": int > 0}``)
- ``GET /bookings`` — list bookings (``?room_id=...`` and
  ``?status=confirmed|cancelled`` filter)
- ``GET /bookings/<id>`` — fetch one booking
- ``POST /bookings/<id>/cancel`` — cancel a booking

A booking's total is ``nights * nightly_rate``, computed by the
server. ``check_out`` must be after ``check_in`` and the guest count
must not exceed the room's capacity. A new booking that overlaps a
confirmed booking for the same room returns 409. Cancelled bookings
free the room again.
"""
from datetime import datetime

from flask import Flask, jsonify, request

STATUSES = ("confirmed", "cancelled")


def _parse_date(value):
    """Parse a YYYY-MM-DD date string, or return None if invalid."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def create_app():
    """Return a Flask app with a fresh, empty in-memory hotel store."""
    app = Flask(__name__)
    app.config["ROOMS"] = {}     # id -> room dict
    app.config["BOOKINGS"] = {}  # id -> booking dict
    app.config["NEXT_ROOM_ID"] = 1
    app.config["NEXT_BOOKING_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def _is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _get_room(room_id):
        room = app.config["ROOMS"].get(room_id)
        if room is None:
            return None, _error(f"room {room_id} not found", 404)
        return room, None

    def _get_booking(booking_id):
        booking = app.config["BOOKINGS"].get(booking_id)
        if booking is None:
            return None, _error(f"booking {booking_id} not found", 404)
        return booking, None

    def _dates_from_query():
        """Validate check_in/check_out query params. Returns (in, out, error)."""
        check_in = _parse_date(request.args.get("check_in"))
        if check_in is None:
            return None, None, _error("check_in is required as YYYY-MM-DD", 400)
        check_out = _parse_date(request.args.get("check_out"))
        if check_out is None:
            return None, None, _error("check_out is required as YYYY-MM-DD", 400)
        if check_out <= check_in:
            return None, None, _error("check_out must be after check_in", 400)
        return check_in, check_out, None

    def _overlaps(room_id, check_in, check_out):
        """True if a confirmed booking for the room overlaps the range."""
        for booking in app.config["BOOKINGS"].values():
            if booking["room_id"] != room_id or booking["status"] != "confirmed":
                continue
            existing_in = _parse_date(booking["check_in"])
            existing_out = _parse_date(booking["check_out"])
            if check_in < existing_out and existing_in < check_out:
                return True
        return False

    @app.post("/rooms")
    def create_room():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        number = data.get("number")
        if not isinstance(number, str) or not number.strip():
            return _error("number is required and must be a non-empty string", 400)
        room_type = data.get("type")
        if not isinstance(room_type, str) or not room_type.strip():
            return _error("type is required and must be a non-empty string", 400)
        nightly_rate = data.get("nightly_rate")
        if not _is_number(nightly_rate) or nightly_rate <= 0:
            return _error("nightly_rate is required and must be a number above 0", 400)
        capacity = data.get("capacity")
        if not _is_int(capacity) or capacity <= 0:
            return _error("capacity is required and must be a positive integer", 400)
        number = number.strip()
        if any(r["number"] == number for r in app.config["ROOMS"].values()):
            return _error(f"room number {number!r} already exists", 409)
        room_id = app.config["NEXT_ROOM_ID"]
        app.config["NEXT_ROOM_ID"] += 1
        room = {
            "id": room_id,
            "number": number,
            "type": room_type.strip(),
            "nightly_rate": nightly_rate,
            "capacity": capacity,
        }
        app.config["ROOMS"][room_id] = room
        return jsonify(room), 201

    @app.get("/rooms")
    def list_rooms():
        rooms = sorted(app.config["ROOMS"].values(), key=lambda r: r["id"])
        return jsonify(rooms)

    @app.get("/rooms/<int:room_id>")
    def get_room(room_id):
        room, err = _get_room(room_id)
        if err is not None:
            return err
        return jsonify(room)

    @app.get("/rooms/<int:room_id>/availability")
    def room_availability(room_id):
        room, err = _get_room(room_id)
        if err is not None:
            return err
        check_in, check_out, date_err = _dates_from_query()
        if date_err is not None:
            return date_err
        return jsonify(
            {
                "room_id": room_id,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "available": not _overlaps(room_id, check_in, check_out),
            }
        )

    @app.get("/availability")
    def availability():
        check_in, check_out, date_err = _dates_from_query()
        if date_err is not None:
            return date_err
        free = [
            room
            for room in sorted(app.config["ROOMS"].values(), key=lambda r: r["id"])
            if not _overlaps(room["id"], check_in, check_out)
        ]
        return jsonify(free)

    @app.post("/bookings")
    def create_booking():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        room_id = data.get("room_id")
        if not _is_int(room_id):
            return _error("room_id is required and must be an integer", 400)
        room, err = _get_room(room_id)
        if err is not None:
            return err
        guest_name = data.get("guest_name")
        if not isinstance(guest_name, str) or not guest_name.strip():
            return _error("guest_name is required and must be a non-empty string", 400)
        check_in = _parse_date(data.get("check_in"))
        if check_in is None:
            return _error("check_in is required as YYYY-MM-DD", 400)
        check_out = _parse_date(data.get("check_out"))
        if check_out is None:
            return _error("check_out is required as YYYY-MM-DD", 400)
        if check_out <= check_in:
            return _error("check_out must be after check_in", 400)
        guests = data.get("guests")
        if not _is_int(guests) or guests <= 0:
            return _error("guests is required and must be a positive integer", 400)
        if guests > room["capacity"]:
            return _error(
                f"room {room['number']} fits {room['capacity']} guests, "
                f"not {guests}",
                400,
            )
        if _overlaps(room_id, check_in, check_out):
            return _error(
                f"room {room['number']} is already booked for those dates", 409
            )
        nights = (check_out - check_in).days
        total = round(nights * room["nightly_rate"], 2)
        booking_id = app.config["NEXT_BOOKING_ID"]
        app.config["NEXT_BOOKING_ID"] += 1
        booking = {
            "id": booking_id,
            "room_id": room_id,
            "guest_name": guest_name.strip(),
            "check_in": check_in.isoformat(),
            "check_out": check_out.isoformat(),
            "guests": guests,
            "nights": nights,
            "total": total,
            "status": "confirmed",
        }
        app.config["BOOKINGS"][booking_id] = booking
        return jsonify(booking), 201

    @app.get("/bookings")
    def list_bookings():
        bookings = sorted(app.config["BOOKINGS"].values(), key=lambda b: b["id"])
        room_id = request.args.get("room_id")
        if room_id is not None:
            try:
                room_id = int(room_id)
            except ValueError:
                return _error("room_id must be an integer", 400)
            bookings = [b for b in bookings if b["room_id"] == room_id]
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error("status must be confirmed or cancelled", 400)
            bookings = [b for b in bookings if b["status"] == status]
        return jsonify(bookings)

    @app.get("/bookings/<int:booking_id>")
    def get_booking(booking_id):
        booking, err = _get_booking(booking_id)
        if err is not None:
            return err
        return jsonify(booking)

    @app.post("/bookings/<int:booking_id>/cancel")
    def cancel_booking(booking_id):
        booking, err = _get_booking(booking_id)
        if err is not None:
            return err
        if booking["status"] == "cancelled":
            return _error(f"booking {booking_id} is already cancelled", 400)
        booking["status"] = "cancelled"
        return jsonify(booking)

    return app
