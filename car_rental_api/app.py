"""Flask car rental API: cars, availability by date range, bookings with double-booking rejection and total price."""
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
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def create_app():
    """Return a Flask app with a fresh, empty in-memory rental store."""
    app = Flask(__name__)
    cars = {}
    bookings = {}
    next_car = [1]
    next_booking = [1]

    def _overlaps(a_start, a_end, b_start, b_end):
        return a_start <= b_end and b_start <= a_end

    def _car_booked(car_id, start, end, ignore_id=None):
        for booking in bookings.values():
            if booking["car_id"] != car_id or booking["status"] != "booked":
                continue
            if ignore_id is not None and booking["id"] == ignore_id:
                continue
            if _overlaps(start, end, _parse_date(booking["start_date"]), _parse_date(booking["end_date"])):
                return True
        return False

    @app.route("/cars", methods=["POST"])
    def create_car():
        data = request.get_json(silent=True) or {}
        make = data.get("make")
        model = data.get("model")
        year = data.get("year")
        rate = data.get("daily_rate")
        if not isinstance(make, str) or not make.strip():
            return _bad("make is required and must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            return _bad("model is required and must be a non-empty string")
        if not _is_int(year) or year < 1900 or year > 2100:
            return _bad("year is required and must be an integer between 1900 and 2100")
        if not _is_number(rate) or rate <= 0:
            return _bad("daily_rate is required and must be a positive number")
        car_id = next_car[0]
        next_car[0] += 1
        cars[car_id] = {
            "id": car_id,
            "make": make.strip(),
            "model": model.strip(),
            "year": year,
            "daily_rate": rate,
        }
        return jsonify(cars[car_id]), 201

    @app.route("/cars", methods=["GET"])
    def list_cars():
        return jsonify(list(cars.values()))

    @app.route("/cars/<int:car_id>", methods=["GET"])
    def get_car(car_id):
        car = cars.get(car_id)
        if car is None:
            return _bad("car not found", 404)
        return jsonify(car)

    @app.route("/cars/available", methods=["GET"])
    def available_cars():
        start = _parse_date(request.args.get("start_date"))
        end = _parse_date(request.args.get("end_date"))
        if start is None or end is None:
            return _bad("start_date and end_date are required as YYYY-MM-DD")
        if end < start:
            return _bad("end_date must not be before start_date")
        free = [c for c in cars.values() if not _car_booked(c["id"], start, end)]
        return jsonify(free)

    @app.route("/bookings", methods=["POST"])
    def create_booking():
        data = request.get_json(silent=True) or {}
        car_id = data.get("car_id")
        customer = data.get("customer_name")
        start = _parse_date(data.get("start_date"))
        end = _parse_date(data.get("end_date"))
        if not _is_int(car_id):
            return _bad("car_id is required and must be an integer")
        if car_id not in cars:
            return _bad("car not found", 404)
        if not isinstance(customer, str) or not customer.strip():
            return _bad("customer_name is required and must be a non-empty string")
        if start is None or end is None:
            return _bad("start_date and end_date are required as YYYY-MM-DD")
        if end < start:
            return _bad("end_date must not be before start_date")
        if _car_booked(car_id, start, end):
            return _bad("car is already booked for part of that date range", 409)
        days = (end - start).days + 1
        booking_id = next_booking[0]
        next_booking[0] += 1
        bookings[booking_id] = {
            "id": booking_id,
            "car_id": car_id,
            "customer_name": customer.strip(),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "days": days,
            "total_price": round(days * cars[car_id]["daily_rate"], 2),
            "status": "booked",
        }
        return jsonify(bookings[booking_id]), 201

    @app.route("/bookings", methods=["GET"])
    def list_bookings():
        result = list(bookings.values())
        car_id = request.args.get("car_id", type=int)
        status = request.args.get("status")
        if car_id is not None:
            result = [b for b in result if b["car_id"] == car_id]
        if status:
            result = [b for b in result if b["status"] == status]
        return jsonify(result)

    @app.route("/bookings/<int:booking_id>", methods=["GET"])
    def get_booking(booking_id):
        booking = bookings.get(booking_id)
        if booking is None:
            return _bad("booking not found", 404)
        return jsonify(booking)

    @app.route("/bookings/<int:booking_id>/cancel", methods=["POST"])
    def cancel_booking(booking_id):
        booking = bookings.get(booking_id)
        if booking is None:
            return _bad("booking not found", 404)
        if booking["status"] == "cancelled":
            return _bad("booking is already cancelled", 409)
        booking["status"] = "cancelled"
        return jsonify(booking)

    @app.route("/cars/<int:car_id>/revenue", methods=["GET"])
    def car_revenue(car_id):
        if car_id not in cars:
            return _bad("car not found", 404)
        mine = [b for b in bookings.values() if b["car_id"] == car_id and b["status"] == "booked"]
        return jsonify({
            "car_id": car_id,
            "booking_count": len(mine),
            "total_revenue": round(sum(b["total_price"] for b in mine), 2),
        })

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "route not found"}), 404

    return app
