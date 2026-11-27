"""Flask job-application tracker API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /applications`` — create an application (``{\"company\":
  str, \"role\": str, \"status\": \"applied\" | \"screening\" |
  \"interview\" | \"offer\" | \"rejected\"?, \"applied_date\":
  \"YYYY-MM-DD\"?, \"interview_date\": \"YYYY-MM-DD\"?, \"salary\":
  number?, \"notes\": str?}``; defaults to ``applied`` with no
  dates)
- ``GET /applications`` — list applications (``?status=`` and
  ``?company=`` filters)
- ``GET /applications/<id>`` — fetch one application
- ``PUT /applications/<id>`` — update company, role, dates, salary
  and/or notes (not status — use the status endpoint)
- ``DELETE /applications/<id>`` — delete an application
- ``POST /applications/<id>/status`` — move an application through
  the pipeline (``{\"status\": ...}``; allowed moves are applied ->
  screening/interview/offer/rejected, screening ->
  interview/offer/rejected, interview -> offer/rejected, offer ->
  rejected (declined); ``rejected`` is terminal and moving
  backwards is a 409, an unknown status is a 400)
- ``GET /stats`` — total applications, counts per status, offer
  rate and interview rate (percentages of the total)
- ``GET /interviews/upcoming`` — applications in ``interview``
  status with an interview date today or later, sorted by date

An application is ``{\"id\": int, \"company\": str, \"role\": str,
\"status\": str, \"applied_date\": str | None, \"interview_date\":
str | None, \"salary\": number | None, \"notes\": str}``.
"""
from datetime import date

from flask import Flask, jsonify, request

STATUSES = ("applied", "screening", "interview", "offer", "rejected")

# Allowed status moves (the hiring pipeline). Anything not listed
# here is rejected with a 409.
ALLOWED_MOVES = {
    "applied": ("screening", "interview", "offer", "rejected"),
    "screening": ("interview", "offer", "rejected"),
    "interview": ("offer", "rejected"),
    "offer": ("rejected",),
    "rejected": (),
}


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_date(value):
    """Return a ``date`` for a ``YYYY-MM-DD`` string, else None."""
    if not isinstance(value, str):
        return None
    try:
        parts = value.split("-")
        if len(parts) != 3 or len(parts[0]) != 4:
            return None
        year, month, day = (int(p) for p in parts)
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def create_app():
    """Return a Flask app with a fresh, empty application store."""
    app = Flask(__name__)
    app.config["APPLICATIONS"] = {}
    app.config["NEXT_APPLICATION_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_application(application_id):
        application = app.config["APPLICATIONS"].get(application_id)
        if application is None:
            return None, _error(
                f"application {application_id} not found", 404
            )
        return application, None

    def _public_application(application):
        return dict(application)

    def _validate_date_field(data, field):
        value = data.get(field)
        if value is not None and _parse_date(value) is None:
            return _error(f"{field} must be a YYYY-MM-DD date or null", 400)
        return None

    @app.post("/applications")
    def create_application():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        company = data.get("company")
        if not isinstance(company, str) or not company.strip():
            return _error(
                "company is required and must be a non-empty string", 400
            )
        role = data.get("role")
        if not isinstance(role, str) or not role.strip():
            return _error(
                "role is required and must be a non-empty string", 400
            )
        status = data.get("status", "applied")
        if status not in STATUSES:
            return _error(f"status must be one of {', '.join(STATUSES)}", 400)
        for field in ("applied_date", "interview_date"):
            err = _validate_date_field(data, field)
            if err is not None:
                return err
        salary = data.get("salary")
        if salary is not None and (not _is_number(salary) or salary < 0):
            return _error("salary must be a number >= 0 or null", 400)
        notes = data.get("notes", "")
        if not isinstance(notes, str):
            return _error("notes must be a string", 400)
        application_id = app.config["NEXT_APPLICATION_ID"]
        app.config["NEXT_APPLICATION_ID"] += 1
        application = {
            "id": application_id,
            "company": company.strip(),
            "role": role.strip(),
            "status": status,
            "applied_date": data.get("applied_date"),
            "interview_date": data.get("interview_date"),
            "salary": salary,
            "notes": notes,
        }
        app.config["APPLICATIONS"][application_id] = application
        return jsonify(_public_application(application)), 201

    @app.get("/applications")
    def list_applications():
        status = request.args.get("status")
        if status is not None and status not in STATUSES:
            return _error(
                f"status filter must be one of {', '.join(STATUSES)}", 400
            )
        company = request.args.get("company")
        applications = sorted(
            app.config["APPLICATIONS"].values(), key=lambda a: a["id"]
        )
        if status is not None:
            applications = [a for a in applications if a["status"] == status]
        if company is not None:
            applications = [
                a for a in applications
                if company.lower() in a["company"].lower()
            ]
        return jsonify([_public_application(a) for a in applications])

    @app.get("/applications/<int:application_id>")
    def get_application(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        return jsonify(_public_application(application))

    @app.put("/applications/<int:application_id>")
    def update_application(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = (
            "company", "role", "applied_date", "interview_date",
            "salary", "notes",
        )
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide company, role, applied_date, "
                "interview_date, salary and/or notes",
                400,
            )
        if "status" in data:
            return _error(
                "use POST /applications/<id>/status to change status", 400
            )
        for field in ("company", "role"):
            if field in data:
                value = data[field]
                if not isinstance(value, str) or not value.strip():
                    return _error(
                        f"{field} must be a non-empty string", 400
                    )
                application[field] = value.strip()
        for field in ("applied_date", "interview_date"):
            if field in data:
                err = _validate_date_field(data, field)
                if err is not None:
                    return err
                application[field] = data[field]
        if "salary" in data:
            salary = data["salary"]
            if salary is not None and (not _is_number(salary) or salary < 0):
                return _error("salary must be a number >= 0 or null", 400)
            application["salary"] = salary
        if "notes" in data:
            if not isinstance(data["notes"], str):
                return _error("notes must be a string", 400)
            application["notes"] = data["notes"]
        return jsonify(_public_application(application))

    @app.delete("/applications/<int:application_id>")
    def delete_application(application_id):
        _application, err = _get_application(application_id)
        if err is not None:
            return err
        del app.config["APPLICATIONS"][application_id]
        return jsonify({"deleted": application_id})

    @app.post("/applications/<int:application_id>/status")
    def change_status(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        new_status = data.get("status")
        if new_status not in STATUSES:
            return _error(
                f"status must be one of {', '.join(STATUSES)}", 400
            )
        if new_status == application["status"]:
            return _error(f"application is already {new_status}", 409)
        if new_status not in ALLOWED_MOVES[application["status"]]:
            return _error(
                f"cannot move an application from "
                f"{application['status']} to {new_status}",
                409,
            )
        application["status"] = new_status
        return jsonify(_public_application(application))

    @app.get("/stats")
    def stats():
        applications = list(app.config["APPLICATIONS"].values())
        total = len(applications)
        by_status = {
            s: sum(1 for a in applications if a["status"] == s)
            for s in STATUSES
        }
        return jsonify(
            {
                "total_applications": total,
                "by_status": by_status,
                "offer_rate": (
                    round(by_status["offer"] / total * 100, 1)
                    if total else 0.0
                ),
                "interview_rate": (
                    round(by_status["interview"] / total * 100, 1)
                    if total else 0.0
                ),
            }
        )

    @app.get("/interviews/upcoming")
    def upcoming_interviews():
        today = date.today()
        upcoming = []
        for application in app.config["APPLICATIONS"].values():
            if application["status"] != "interview":
                continue
            interview_date = application.get("interview_date")
            parsed = _parse_date(interview_date) if interview_date else None
            if parsed is not None and parsed >= today:
                upcoming.append((parsed, application))
        upcoming.sort(key=lambda item: (item[0], item[1]["id"]))
        return jsonify(
            [_public_application(a) for _d, a in upcoming]
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
