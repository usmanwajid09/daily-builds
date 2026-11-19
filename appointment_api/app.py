"""Flask appointment booking API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /services`` — create a service (``{"name": str,
  "duration_minutes": int > 0, "price": number >= 0?}``)
- ``GET /services`` — list services
- ``GET /services/<id>`` — fetch one service
- ``POST /appointments`` — book an appointment (``{"service_id": int,
  "customer_name": str, "date": "YYYY-MM-DD",
  "start_time": "HH:MM"}``). The end time is computed from the
  service's duration. A new booking is rejected with 409 if its time
  range overlaps another booked appointment for the same service on
  the same date; cancelled appointments do not block a slot.
- ``GET /appointments`` — list appointments; ``?date=``,
  ``?service_id=`` and ``?status=`` filters
- ``GET /appointments/upcoming`` — booked appointments from today
  onwards, earliest first
- ``GET /appointments/<id>`` — fetch one appointment
- ``POST /appointments/<id>/cancel`` — cancel a booked appointment
- ``DELETE /appointments/<id>`` — delete an appointment

An appointment is ``{"id": int, "service_id": int,
"customer_name": str, "date": str, "start_time": str, "end_time": str,
"status": "booked" | "cancelled"}``.
"""
from datetime import date, datetime, time, timedelta

from flask import Flask, jsonify, request

STATUSES = ("booked", "cancelled")


def _parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _parse_time(value):
    try:
        return datetime.strptime(value, "%H:%M").time()
    except (TypeError, ValueError):
        return None


def _add_minutes(start, minutes):
    end = datetime.combine(date(2000, 1, 1), start) + timedelta(minutes=minutes)
    return end.time()


def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def create_app():
    """Return a Flask app with a fresh, empty in-memory booking store."""
    app = Flask(__name__)
    app.config["SERVICES"] = {}
    app.config["APPOINTMENTS"] = {}
    app.config["NEXT_SERVICE_ID"] = 1
    app.config["NEXT_APPOINTMENT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_service(service_id):
        service = app.config["SERVICES"].get(service_id)
        if service is None:
            return None, _error(f"service {service_id} not found", 404)
        return service, None

    def _get_appointment(appointment_id):
        appt = app.config["APPOINTMENTS"].get(appointment_id)
        if appt is None:
            return None, _error(f"appointment {appointment_id} not found", 404)
        return appt, None

    @app.post("/services")
    def create_service():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        duration = data.get("duration_minutes")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration <= 0
        ):
            return _error("duration_minutes must be a positive integer", 400)
        price = data.get("price", 0)
        if isinstance(price, bool) or not isinstance(price, (int, float)) or price < 0:
            return _error("price must be a number >= 0", 400)
        service_id = app.config["NEXT_SERVICE_ID"]
        app.config["NEXT_SERVICE_ID"] += 1
        service = {
            "id": service_id,
            "name": name.strip(),
            "duration_minutes": duration,
            "price": price,
        }
        app.config["SERVICES"][service_id] = service
        return jsonify(service), 201

    @app.get("/services")
    def list_services():
        services = sorted(app.config["SERVICES"].values(), key=lambda s: s["id"])
        return jsonify(services)

    @app.get("/services/<int:service_id>")
    def get_service(service_id):
        service, err = _get_service(service_id)
        if err is not None:
            return err
        return jsonify(service)

    @app.post("/appointments")
    def book_appointment():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        service_id = data.get("service_id")
        if isinstance(service_id, bool) or not isinstance(service_id, int):
            return _error("service_id must be an integer", 400)
        service, err = _get_service(service_id)
        if err is not None:
            return err
        customer = data.get("customer_name")
        if not isinstance(customer, str) or not customer.strip():
            return _error(
                "customer_name is required and must be a non-empty string", 400
            )
        day = _parse_date(data.get("date"))
        if day is None:
            return _error("date must be in YYYY-MM-DD format", 400)
        start = _parse_time(data.get("start_time"))
        if start is None:
            return _error("start_time must be in HH:MM format", 400)
        end = _add_minutes(start, service["duration_minutes"])
        if end <= start:
            return _error("appointment must end on the same day it starts", 400)
        for other in app.config["APPOINTMENTS"].values():
            if (
                other["service_id"] == service_id
                and other["date"] == data["date"]
                and other["status"] == "booked"
                and _overlaps(
                    start, end,
                    _parse_time(other["start_time"]), _parse_time(other["end_time"]),
                )
            ):
                return _error(
                    f"slot {data['start_time']}-{end.strftime('%H:%M')} on "
                    f"{data['date']} overlaps appointment #{other['id']} "
                    f"({other['start_time']}-{other['end_time']})",
                    409,
                )
        appointment_id = app.config["NEXT_APPOINTMENT_ID"]
        app.config["NEXT_APPOINTMENT_ID"] += 1
        appointment = {
            "id": appointment_id,
            "service_id": service_id,
            "customer_name": customer.strip(),
            "date": data["date"],
            "start_time": start.strftime("%H:%M"),
            "end_time": end.strftime("%H:%M"),
            "status": "booked",
        }
        app.config["APPOINTMENTS"][appointment_id] = appointment
        return jsonify(appointment), 201

    @app.get("/appointments")
    def list_appointments():
        appointments = sorted(
            app.config["APPOINTMENTS"].values(),
            key=lambda a: (a["date"], a["start_time"], a["id"]),
        )
        day = request.args.get("date")
        if day is not None:
            if _parse_date(day) is None:
                return _error("date must be in YYYY-MM-DD format", 400)
            appointments = [a for a in appointments if a["date"] == day]
        service_id = request.args.get("service_id")
        if service_id is not None:
            try:
                sid = int(service_id)
            except ValueError:
                return _error("service_id must be an integer", 400)
            appointments = [a for a in appointments if a["service_id"] == sid]
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error(f"status must be one of {', '.join(STATUSES)}", 400)
            appointments = [a for a in appointments if a["status"] == status]
        return jsonify(appointments)

    @app.get("/appointments/upcoming")
    def list_upcoming():
        today = date.today().isoformat()
        upcoming = [
            a for a in app.config["APPOINTMENTS"].values()
            if a["status"] == "booked" and a["date"] >= today
        ]
        upcoming.sort(key=lambda a: (a["date"], a["start_time"], a["id"]))
        return jsonify(upcoming)

    @app.get("/appointments/<int:appointment_id>")
    def get_appointment(appointment_id):
        appt, err = _get_appointment(appointment_id)
        if err is not None:
            return err
        return jsonify(appt)

    @app.post("/appointments/<int:appointment_id>/cancel")
    def cancel_appointment(appointment_id):
        appt, err = _get_appointment(appointment_id)
        if err is not None:
            return err
        if appt["status"] == "cancelled":
            return _error(f"appointment {appointment_id} is already cancelled", 400)
        appt["status"] = "cancelled"
        return jsonify(appt)

    @app.delete("/appointments/<int:appointment_id>")
    def delete_appointment(appointment_id):
        _appt, err = _get_appointment(appointment_id)
        if err is not None:
            return err
        del app.config["APPOINTMENTS"][appointment_id]
        return jsonify({"deleted": appointment_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
