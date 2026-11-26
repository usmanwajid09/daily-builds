"""Flask leave tracker API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /employees`` — create an employee (``{"name": str,
  "balances": {"annual": int, "sick": int, "casual": int}?}``;
  defaults are 20 annual, 10 sick, 7 casual days)
- ``GET /employees`` — list all employees with remaining balances
- ``GET /employees/<id>`` — fetch one employee
- ``GET /employees/<id>/balance`` — remaining leave balance per type
- ``POST /employees/<id>/leave`` — request leave (``{"type":
  "annual" | "sick" | "casual", "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD", "reason": str?}``); days are counted
  inclusively
- ``GET /employees/<id>/leave`` — list an employee's requests
- ``GET /leave/<id>`` — fetch one request
- ``POST /leave/<id>/approve`` — approve a pending request (deducts
  the days from the balance)
- ``POST /leave/<id>/reject`` — reject a pending request

A request overlapping an existing pending or approved request returns
``409``; a request for more days than the remaining balance returns
``400``; deciding an already-decided request returns ``409``.
"""
from datetime import date

from flask import Flask, jsonify, request

LEAVE_TYPES = ("annual", "sick", "casual")
DEFAULT_BALANCES = {"annual": 20, "sick": 10, "casual": 7}


def create_app():
    """Return a Flask app with a fresh, empty in-memory leave store."""
    app = Flask(__name__)
    app.config["EMPLOYEES"] = {}
    app.config["REQUESTS"] = {}
    app.config["NEXT_EMPLOYEE_ID"] = 1
    app.config["NEXT_REQUEST_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_date(value):
        """Return a date object for a YYYY-MM-DD string, else None."""
        if not isinstance(value, str):
            return None
        try:
            parts = value.split("-")
            if len(parts) != 3 or len(parts[0]) != 4:
                return None
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, TypeError):
            return None

    def _get_employee(employee_id):
        employee = app.config["EMPLOYEES"].get(employee_id)
        if employee is None:
            return None, _error(f"employee {employee_id} not found", 404)
        return employee, None

    def _public_employee(employee):
        return {
            "id": employee["id"],
            "name": employee["name"],
            "balances": dict(employee["balances"]),
        }

    def _public_request(req):
        return {
            "id": req["id"],
            "employee_id": req["employee_id"],
            "type": req["type"],
            "start_date": req["start_date"],
            "end_date": req["end_date"],
            "days": req["days"],
            "reason": req["reason"],
            "status": req["status"],
        }

    @app.post("/employees")
    def create_employee():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        balances = dict(DEFAULT_BALANCES)
        supplied = data.get("balances")
        if supplied is not None:
            if not isinstance(supplied, dict):
                return _error("balances must be an object", 400)
            for leave_type, days in supplied.items():
                if leave_type not in LEAVE_TYPES:
                    return _error(
                        "balances keys must be one of: " + ", ".join(LEAVE_TYPES), 400
                    )
                if isinstance(days, bool) or not isinstance(days, int) or days < 0:
                    return _error(
                        f"balance for {leave_type} must be a non-negative integer", 400
                    )
                balances[leave_type] = days
        employee_id = app.config["NEXT_EMPLOYEE_ID"]
        app.config["NEXT_EMPLOYEE_ID"] += 1
        employee = {"id": employee_id, "name": name.strip(), "balances": balances}
        app.config["EMPLOYEES"][employee_id] = employee
        return jsonify(_public_employee(employee)), 201

    @app.get("/employees")
    def list_employees():
        employees = sorted(app.config["EMPLOYEES"].values(), key=lambda e: e["id"])
        return jsonify([_public_employee(e) for e in employees])

    @app.get("/employees/<int:employee_id>")
    def get_employee(employee_id):
        employee, err = _get_employee(employee_id)
        if err is not None:
            return err
        return jsonify(_public_employee(employee))

    @app.get("/employees/<int:employee_id>/balance")
    def get_balance(employee_id):
        employee, err = _get_employee(employee_id)
        if err is not None:
            return err
        return jsonify(
            {"employee_id": employee["id"], "balances": dict(employee["balances"])}
        )

    @app.post("/employees/<int:employee_id>/leave")
    def request_leave(employee_id):
        employee, err = _get_employee(employee_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        leave_type = data.get("type")
        if leave_type not in LEAVE_TYPES:
            return _error(
                "type is required and must be one of: " + ", ".join(LEAVE_TYPES), 400
            )
        start = _parse_date(data.get("start_date"))
        end = _parse_date(data.get("end_date"))
        if start is None or end is None:
            return _error(
                "start_date and end_date are required and must be valid "
                "YYYY-MM-DD strings",
                400,
            )
        if end < start:
            return _error("end_date cannot be before start_date", 400)
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            return _error("reason must be a string", 400)
        days = (end - start).days + 1
        if days > employee["balances"][leave_type]:
            return _error(
                f"insufficient {leave_type} leave balance: requested {days} day(s), "
                f"{employee['balances'][leave_type]} remaining",
                400,
            )
        for other in app.config["REQUESTS"].values():
            if other["employee_id"] != employee_id:
                continue
            if other["status"] not in ("pending", "approved"):
                continue
            other_start = _parse_date(other["start_date"])
            other_end = _parse_date(other["end_date"])
            if start <= other_end and end >= other_start:
                return _error(
                    f"leave request overlaps request {other['id']} "
                    f"({other['start_date']} to {other['end_date']})",
                    409,
                )
        request_id = app.config["NEXT_REQUEST_ID"]
        app.config["NEXT_REQUEST_ID"] += 1
        req = {
            "id": request_id,
            "employee_id": employee_id,
            "type": leave_type,
            "start_date": data["start_date"],
            "end_date": data["end_date"],
            "days": days,
            "reason": reason.strip(),
            "status": "pending",
        }
        app.config["REQUESTS"][request_id] = req
        return jsonify(_public_request(req)), 201

    @app.get("/employees/<int:employee_id>/leave")
    def list_leave(employee_id):
        employee, err = _get_employee(employee_id)
        if err is not None:
            return err
        requests = [
            r for r in app.config["REQUESTS"].values() if r["employee_id"] == employee_id
        ]
        requests.sort(key=lambda r: r["id"])
        return jsonify([_public_request(r) for r in requests])

    @app.get("/leave/<int:request_id>")
    def get_request(request_id):
        req = app.config["REQUESTS"].get(request_id)
        if req is None:
            return _error(f"leave request {request_id} not found", 404)
        return jsonify(_public_request(req))

    def _decide(request_id, decision):
        req = app.config["REQUESTS"].get(request_id)
        if req is None:
            return _error(f"leave request {request_id} not found", 404)
        if req["status"] != "pending":
            return _error(
                f"leave request {request_id} has already been {req['status']}", 409
            )
        if decision == "approved":
            employee = app.config["EMPLOYEES"][req["employee_id"]]
            if req["days"] > employee["balances"][req["type"]]:
                return _error(
                    f"insufficient {req['type']} leave balance: requested "
                    f"{req['days']} day(s), "
                    f"{employee['balances'][req['type']]} remaining",
                    400,
                )
            employee["balances"][req["type"]] -= req["days"]
        req["status"] = decision
        return jsonify(_public_request(req))

    @app.post("/leave/<int:request_id>/approve")
    def approve_request(request_id):
        return _decide(request_id, "approved")

    @app.post("/leave/<int:request_id>/reject")
    def reject_request(request_id):
        return _decide(request_id, "rejected")

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
