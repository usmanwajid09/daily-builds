"""Flask attendance tracker API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /students`` — create a student (``{"name": str}``)
- ``GET /students`` — list all students with their marks
- ``GET /students/<id>`` — fetch one student
- ``POST /students/<id>/marks`` — mark a day (``{"date": "YYYY-MM-DD",
  "status": "present" | "absent" | "late"}``)
- ``GET /students/<id>/attendance`` — per-student report with percentage
- ``GET /summary`` — class-wide summary with overall percentage

Attendance percentage counts ``present`` and ``late`` as attended days
over the total marked days. Marking the same date twice returns ``409``;
an unknown student returns ``404``; an invalid status or date
returns ``400``.
"""
from datetime import date

from flask import Flask, jsonify, request

STATUSES = ("present", "absent", "late")


def create_app():
    """Return a Flask app with a fresh, empty in-memory attendance store."""
    app = Flask(__name__)
    app.config["STUDENTS"] = {}
    app.config["NEXT_ID"] = 1

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

    def _get_student(student_id):
        student = app.config["STUDENTS"].get(student_id)
        if student is None:
            return None, _error(f"student {student_id} not found", 404)
        return student, None

    def _percentage(student):
        total = len(student["marks"])
        if total == 0:
            return 0.0
        attended = sum(
            1 for status in student["marks"].values() if status in ("present", "late")
        )
        return round(attended / total * 100, 2)

    def _public(student):
        return {
            "id": student["id"],
            "name": student["name"],
            "marks": dict(sorted(student["marks"].items())),
            "total_marks": len(student["marks"]),
            "attendance_percentage": _percentage(student),
        }

    @app.post("/students")
    def create_student():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        student_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        student = {"id": student_id, "name": name.strip(), "marks": {}}
        app.config["STUDENTS"][student_id] = student
        return jsonify(_public(student)), 201

    @app.get("/students")
    def list_students():
        students = sorted(app.config["STUDENTS"].values(), key=lambda s: s["id"])
        return jsonify([_public(s) for s in students])

    @app.get("/students/<int:student_id>")
    def get_student(student_id):
        student, err = _get_student(student_id)
        if err is not None:
            return err
        return jsonify(_public(student))

    @app.post("/students/<int:student_id>/marks")
    def mark_attendance(student_id):
        student, err = _get_student(student_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        raw = data.get("date")
        if _parse_date(raw) is None:
            return _error("date is required and must be a valid YYYY-MM-DD string", 400)
        status = data.get("status")
        if status not in STATUSES:
            return _error(
                "status is required and must be one of: " + ", ".join(STATUSES), 400
            )
        if raw in student["marks"]:
            return _error(
                f"student {student_id} already has a mark for {raw}", 409
            )
        student["marks"][raw] = status
        return jsonify(_public(student)), 201

    @app.get("/students/<int:student_id>/attendance")
    def student_attendance(student_id):
        student, err = _get_student(student_id)
        if err is not None:
            return err
        marks = student["marks"].values()
        return jsonify(
            {
                "student_id": student["id"],
                "name": student["name"],
                "total_days": len(student["marks"]),
                "present": sum(1 for m in marks if m == "present"),
                "absent": sum(1 for m in student["marks"].values() if m == "absent"),
                "late": sum(1 for m in student["marks"].values() if m == "late"),
                "attendance_percentage": _percentage(student),
            }
        )

    @app.get("/summary")
    def class_summary():
        students = sorted(app.config["STUDENTS"].values(), key=lambda s: s["id"])
        total_marks = sum(len(s["marks"]) for s in students)
        attended = sum(
            1
            for s in students
            for status in s["marks"].values()
            if status in ("present", "late")
        )
        overall = round(attended / total_marks * 100, 2) if total_marks else 0.0
        return jsonify(
            {
                "total_students": len(students),
                "total_marks": total_marks,
                "overall_attendance_percentage": overall,
                "students": [
                    {
                        "id": s["id"],
                        "name": s["name"],
                        "attendance_percentage": _percentage(s),
                    }
                    for s in students
                ],
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
