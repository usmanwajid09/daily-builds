"""Flask movie ticket booking API.

In-memory store, fresh for each app instance. All responses are JSON.

Movies — ``POST /movies`` (``{"title": str, "duration_minutes": int,
"genre": str?}``); ``GET /movies``; ``GET /movies/<id>``;
``DELETE /movies/<id>`` (only while it has no screenings).

Screenings — ``POST /screenings`` (``{"movie_id": int, "screen": str,
"start_time": str, "rows": int, "seats_per_row": int,
"ticket_price": number}``); ``GET /screenings``;
``GET /screenings/<id>`` (includes the full seat map);
``GET /screenings/<id>/bookings``.

A screening's seats are labelled ``A1``, ``A2``, ... row letters from A
up to ``rows`` rows, seat numbers 1 to ``seats_per_row``. A screening is a
dict with ``seats`` mapping each label to ``"available"`` or ``"booked"``.

Bookings — ``POST /screenings/<id>/book`` (``{"customer_name": str,
"seats": [str, ...]}``); ``GET /bookings/<id>``;
``POST /bookings/<id>/cancel``; ``GET /bookings``.

Booking is atomic: if any requested seat is invalid, already booked, or
duplicated in the request, nothing is booked and the API returns 400 or
409. The booking's ``total_price`` is ``len(seats) * ticket_price``.
Cancelling a booking frees its seats.
"""
import re

from flask import Flask, jsonify, request

SEAT_RE = re.compile(r"^([A-Z])(\d+)$")


def _seat_labels(rows, seats_per_row):
    labels = []
    for row in range(rows):
        letter = chr(ord("A") + row)
        for number in range(1, seats_per_row + 1):
            labels.append(f"{letter}{number}")
    return labels


def create_app():
    """Return a Flask app with a fresh, empty in-memory booking store."""
    app = Flask(__name__)
    app.config["MOVIES"] = {}
    app.config["SCREENINGS"] = {}
    app.config["BOOKINGS"] = {}
    app.config["NEXT_MOVIE_ID"] = 1
    app.config["NEXT_SCREENING_ID"] = 1
    app.config["NEXT_BOOKING_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_movie(movie_id):
        movie = app.config["MOVIES"].get(movie_id)
        if movie is None:
            return None, _error(f"movie {movie_id} not found", 404)
        return movie, None

    def _get_screening(screening_id):
        screening = app.config["SCREENINGS"].get(screening_id)
        if screening is None:
            return None, _error(f"screening {screening_id} not found", 404)
        return screening, None

    def _get_booking(booking_id):
        booking = app.config["BOOKINGS"].get(booking_id)
        if booking is None:
            return None, _error(f"booking {booking_id} not found", 404)
        return booking, None

    def _public_screening(screening):
        return {
            "id": screening["id"],
            "movie_id": screening["movie_id"],
            "screen": screening["screen"],
            "start_time": screening["start_time"],
            "rows": screening["rows"],
            "seats_per_row": screening["seats_per_row"],
            "ticket_price": screening["ticket_price"],
            "seats": dict(screening["seats"]),
            "available_seats": sum(
                1 for state in screening["seats"].values() if state == "available"
            ),
            "booked_seats": sum(
                1 for state in screening["seats"].values() if state == "booked"
            ),
        }

    @app.post("/movies")
    def create_movie():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        duration = data.get("duration_minutes")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration <= 0
        ):
            return _error("duration_minutes must be a positive integer", 400)
        genre = data.get("genre")
        if genre is not None and (not isinstance(genre, str) or not genre.strip()):
            return _error("genre must be a non-empty string", 400)
        movie_id = app.config["NEXT_MOVIE_ID"]
        app.config["NEXT_MOVIE_ID"] += 1
        movie = {
            "id": movie_id,
            "title": title.strip(),
            "duration_minutes": duration,
            "genre": genre.strip() if genre else None,
        }
        app.config["MOVIES"][movie_id] = movie
        return jsonify(movie), 201

    @app.get("/movies")
    def list_movies():
        return jsonify(list(app.config["MOVIES"].values()))

    @app.get("/movies/<int:movie_id>")
    def get_movie(movie_id):
        movie, err = _get_movie(movie_id)
        if err is not None:
            return err
        return jsonify(movie)

    @app.delete("/movies/<int:movie_id>")
    def delete_movie(movie_id):
        movie, err = _get_movie(movie_id)
        if err is not None:
            return err
        if any(s["movie_id"] == movie_id for s in app.config["SCREENINGS"].values()):
            return _error("cannot delete a movie that has screenings", 409)
        del app.config["MOVIES"][movie_id]
        return jsonify({"deleted": movie_id})

    @app.post("/screenings")
    def create_screening():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        movie_id = data.get("movie_id")
        if isinstance(movie_id, bool) or not isinstance(movie_id, int):
            return _error("movie_id must be an integer", 400)
        movie, err = _get_movie(movie_id)
        if err is not None:
            return err
        screen = data.get("screen")
        if not isinstance(screen, str) or not screen.strip():
            return _error("screen is required and must be a non-empty string", 400)
        start_time = data.get("start_time")
        if not isinstance(start_time, str) or not start_time.strip():
            return _error(
                "start_time is required and must be a non-empty string", 400
            )
        rows = data.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 26:
            return _error("rows must be an integer between 1 and 26", 400)
        seats_per_row = data.get("seats_per_row")
        if (
            isinstance(seats_per_row, bool)
            or not isinstance(seats_per_row, int)
            or not 1 <= seats_per_row <= 50
        ):
            return _error("seats_per_row must be an integer between 1 and 50", 400)
        ticket_price = data.get("ticket_price")
        if (
            isinstance(ticket_price, bool)
            or not isinstance(ticket_price, (int, float))
            or ticket_price <= 0
        ):
            return _error("ticket_price must be a positive number", 400)
        screening_id = app.config["NEXT_SCREENING_ID"]
        app.config["NEXT_SCREENING_ID"] += 1
        screening = {
            "id": screening_id,
            "movie_id": movie_id,
            "screen": screen.strip(),
            "start_time": start_time.strip(),
            "rows": rows,
            "seats_per_row": seats_per_row,
            "ticket_price": round(float(ticket_price), 2),
            "seats": {label: "available" for label in _seat_labels(rows, seats_per_row)},
        }
        app.config["SCREENINGS"][screening_id] = screening
        return jsonify(_public_screening(screening)), 201

    @app.get("/screenings")
    def list_screenings():
        screenings = list(app.config["SCREENINGS"].values())
        movie_id = request.args.get("movie_id")
        if movie_id is not None:
            if not movie_id.isdigit():
                return _error("movie_id filter must be an integer", 400)
            screenings = [s for s in screenings if s["movie_id"] == int(movie_id)]
        return jsonify([_public_screening(s) for s in screenings])

    @app.get("/screenings/<int:screening_id>")
    def get_screening(screening_id):
        screening, err = _get_screening(screening_id)
        if err is not None:
            return err
        return jsonify(_public_screening(screening))

    @app.post("/screenings/<int:screening_id>/book")
    def book_seats(screening_id):
        screening, err = _get_screening(screening_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        customer_name = data.get("customer_name")
        if not isinstance(customer_name, str) or not customer_name.strip():
            return _error(
                "customer_name is required and must be a non-empty string", 400
            )
        seats = data.get("seats")
        if not isinstance(seats, list) or not seats:
            return _error("seats must be a non-empty list of seat labels", 400)
        if len(set(seats)) != len(seats):
            return _error("seats must not contain duplicates", 400)
        for seat in seats:
            if not isinstance(seat, str) or not SEAT_RE.match(seat):
                return _error(f"seat {seat!r} is not a valid seat label", 400)
            if seat not in screening["seats"]:
                return _error(f"seat {seat} does not exist for this screening", 400)
            if screening["seats"][seat] == "booked":
                return _error(f"seat {seat} is already booked", 409)
        booking_id = app.config["NEXT_BOOKING_ID"]
        app.config["NEXT_BOOKING_ID"] += 1
        for seat in seats:
            screening["seats"][seat] = "booked"
        booking = {
            "id": booking_id,
            "screening_id": screening_id,
            "customer_name": customer_name.strip(),
            "seats": list(seats),
            "total_price": round(len(seats) * screening["ticket_price"], 2),
            "status": "active",
        }
        app.config["BOOKINGS"][booking_id] = booking
        return jsonify(booking), 201

    @app.get("/screenings/<int:screening_id>/bookings")
    def list_screening_bookings(screening_id):
        screening, err = _get_screening(screening_id)
        if err is not None:
            return err
        return jsonify(
            [
                b
                for b in app.config["BOOKINGS"].values()
                if b["screening_id"] == screening_id
            ]
        )

    @app.get("/bookings")
    def list_bookings():
        bookings = list(app.config["BOOKINGS"].values())
        status = request.args.get("status")
        if status is not None:
            if status not in ("active", "cancelled"):
                return _error("status filter must be active or cancelled", 400)
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
            return _error("booking is already cancelled", 409)
        screening = app.config["SCREENINGS"][booking["screening_id"]]
        for seat in booking["seats"]:
            screening["seats"][seat] = "available"
        booking["status"] = "cancelled"
        return jsonify(booking)

    return app
