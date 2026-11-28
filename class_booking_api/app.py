"""Flask gym class booking API.

In-memory store only: classes, their bookings, and a waitlist per class.
When a class is full, new members join the waitlist; cancelling a booked
member promotes the first waitlisted member into the freed spot.
"""

import re
from datetime import datetime

from flask import Flask, jsonify, request

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def create_app():
    """Return a fresh Flask app with an empty in-memory store."""
    app = Flask(__name__)
    app.config["classes"] = {}
    app.config["next_id"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_class(class_id):
        return app.config["classes"].get(class_id)

    def _public(cls):
        return {
            "id": cls["id"],
            "name": cls["name"],
            "instructor": cls["instructor"],
            "date": cls["date"],
            "start_time": cls["start_time"],
            "capacity": cls["capacity"],
            "booked_count": len(cls["bookings"]),
            "waitlist_count": len(cls["waitlist"]),
            "spots_left": cls["capacity"] - len(cls["bookings"]),
        }

    def _valid_date(value):
        if not isinstance(value, str) or not _DATE_RE.match(value):
            return False
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return False
        return True

    def _valid_time(value):
        if not isinstance(value, str) or not _TIME_RE.match(value):
            return False
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError:
            return False
        return True

    @app.errorhandler(404)
    def _not_found(_err):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(405)
    def _method_not_allowed(_err):
        return jsonify({"error": "method not allowed"}), 405

    @app.post("/classes")
    def create_class():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required", 400)
        if not _valid_date(data.get("date")):
            return _error("date must be a valid date in YYYY-MM-DD format", 400)
        if not _valid_time(data.get("start_time")):
            return _error("start_time must be a valid time in HH:MM format", 400)
        capacity = data.get("capacity")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            return _error("capacity must be an integer greater than 0", 400)
        instructor = data.get("instructor", "")
        if instructor is None:
            instructor = ""
        if not isinstance(instructor, str):
            return _error("instructor must be a string", 400)

        class_id = app.config["next_id"]
        app.config["next_id"] += 1
        cls = {
            "id": class_id,
            "name": name.strip(),
            "instructor": instructor.strip(),
            "date": data["date"],
            "start_time": data["start_time"],
            "capacity": capacity,
            "bookings": [],
            "waitlist": [],
        }
        app.config["classes"][class_id] = cls
        return jsonify(_public(cls)), 201

    @app.get("/classes")
    def list_classes():
        return jsonify([_public(cls) for cls in app.config["classes"].values()])

    @app.get("/classes/<int:class_id>")
    def get_class(class_id):
        cls = _get_class(class_id)
        if cls is None:
            return _error("class not found", 404)
        return jsonify(_public(cls))

    @app.delete("/classes/<int:class_id>")
    def delete_class(class_id):
        cls = _get_class(class_id)
        if cls is None:
            return _error("class not found", 404)
        del app.config["classes"][class_id]
        return jsonify({"message": "class deleted", "id": class_id})

    @app.post("/classes/<int:class_id>/book")
    def book_class(class_id):
        cls = _get_class(class_id)
        if cls is None:
            return _error("class not found", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        member_name = data.get("member_name")
        if not isinstance(member_name, str) or not member_name.strip():
            return _error("member_name is required", 400)
        member_name = member_name.strip()
        if member_name in cls["bookings"] or member_name in cls["waitlist"]:
            return _error("member already booked or waitlisted for this class", 409)
        if len(cls["bookings"]) < cls["capacity"]:
            cls["bookings"].append(member_name)
            return jsonify({
                "status": "booked",
                "class_id": class_id,
                "member_name": member_name,
                "spots_left": cls["capacity"] - len(cls["bookings"]),
            }), 201
        cls["waitlist"].append(member_name)
        return jsonify({
            "status": "waitlisted",
            "class_id": class_id,
            "member_name": member_name,
            "waitlist_position": len(cls["waitlist"]),
        }), 201

    @app.post("/classes/<int:class_id>/cancel")
    def cancel_booking(class_id):
        cls = _get_class(class_id)
        if cls is None:
            return _error("class not found", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        member_name = data.get("member_name")
        if not isinstance(member_name, str) or not member_name.strip():
            return _error("member_name is required", 400)
        member_name = member_name.strip()
        promoted = None
        if member_name in cls["bookings"]:
            cls["bookings"].remove(member_name)
            if cls["waitlist"]:
                promoted = cls["waitlist"].pop(0)
                cls["bookings"].append(promoted)
        elif member_name in cls["waitlist"]:
            cls["waitlist"].remove(member_name)
        else:
            return _error("member not found in bookings or waitlist", 404)
        return jsonify({
            "status": "cancelled",
            "class_id": class_id,
            "member_name": member_name,
            "promoted": promoted,
            "spots_left": cls["capacity"] - len(cls["bookings"]),
        })

    @app.get("/classes/<int:class_id>/bookings")
    def list_bookings(class_id):
        cls = _get_class(class_id)
        if cls is None:
            return _error("class not found", 404)
        return jsonify([
            {"member_name": name, "status": "booked"} for name in cls["bookings"]
        ])

    @app.get("/classes/<int:class_id>/waitlist")
    def list_waitlist(class_id):
        cls = _get_class(class_id)
        if cls is None:
            return _error("class not found", 404)
        return jsonify([
            {"member_name": name, "position": i + 1}
            for i, name in enumerate(cls["waitlist"])
        ])

    return app
