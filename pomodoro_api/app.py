"""Flask pomodoro timer API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /sessions`` — create a session (``{\"task\": str,
  \"type\": \"focus\" | \"short_break\" | \"long_break\"?,
  \"duration_minutes\": int?}``; defaults: focus, 25 min for focus,
  5 for short break, 15 for long break)
- ``GET /sessions`` — list sessions; ``?status=`` and ``?type=``
  filters, ``?date=YYYY-MM-DD`` filters by creation date
- ``GET /sessions/<id>`` — fetch one session
- ``POST /sessions/<id>/start`` — start a pending session
  (409 if already started / finished / cancelled)
- ``POST /sessions/<id>/complete`` — complete a running session
  (409 unless running)
- ``POST /sessions/<id>/cancel`` — cancel a pending or running
  session (409 if already completed / cancelled)
- ``DELETE /sessions/<id>`` — delete a session
- ``GET /stats`` — totals, counts by type and status, completed
  focus minutes, today's counts
- ``GET /stats/daily?date=YYYY-MM-DD`` — stats for one day

A session is ``{\"id\": int, \"task\": str, \"type\": str,
\"duration_minutes\": int, \"status\": \"pending\" | \"running\" |
\"completed\" | \"cancelled\", \"started_at\": str | None,
\"completed_at\": str | None, \"created_at\": str}``. Timestamps are
UTC ISO 8601 strings.
"""
from datetime import datetime, timezone

from flask import Flask, jsonify, request

TYPES = ("focus", "short_break", "long_break")
STATUSES = ("pending", "running", "completed", "cancelled")
DEFAULT_DURATIONS = {"focus": 25, "short_break": 5, "long_break": 15}
MIN_DURATION = 1
MAX_DURATION = 240


def _now():
    return datetime.now(timezone.utc)


def _iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory session store."""
    app = Flask(__name__)
    app.config["SESSIONS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _public(session):
        return dict(session)

    def _get_session(session_id):
        session = app.config["SESSIONS"].get(session_id)
        if session is None:
            return None, _error(f"session {session_id} not found", 404)
        return session, None

    def _parse_date(value, name):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

    def _daily_stats(day):
        sessions = [
            s for s in app.config["SESSIONS"].values()
            if s["created_at"][:10] == day.isoformat()
        ]
        completed = [s for s in sessions if s["status"] == "completed"]
        focus_minutes = sum(
            s["duration_minutes"] for s in completed if s["type"] == "focus"
        )
        return {
            "date": day.isoformat(),
            "total_sessions": len(sessions),
            "completed_sessions": len(completed),
            "by_type": {
                t: sum(1 for s in sessions if s["type"] == t) for t in TYPES
            },
            "by_status": {
                st: sum(1 for s in sessions if s["status"] == st)
                for st in STATUSES
            },
            "focus_minutes_completed": focus_minutes,
        }

    @app.post("/sessions")
    def create_session():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        task = data.get("task")
        if not isinstance(task, str) or not task.strip():
            return _error("task is required and must be a non-empty string", 400)
        session_type = data.get("type", "focus")
        if session_type not in TYPES:
            return _error(
                f"type must be one of {', '.join(TYPES)}", 400
            )
        duration = data.get("duration_minutes",
                            DEFAULT_DURATIONS[session_type])
        if not _is_int(duration) or not MIN_DURATION <= duration <= MAX_DURATION:
            return _error(
                f"duration_minutes must be an integer between "
                f"{MIN_DURATION} and {MAX_DURATION}",
                400,
            )
        session_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        session = {
            "id": session_id,
            "task": task.strip(),
            "type": session_type,
            "duration_minutes": duration,
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "created_at": _iso(_now()),
        }
        app.config["SESSIONS"][session_id] = session
        return jsonify(_public(session)), 201

    @app.get("/sessions")
    def list_sessions():
        sessions = sorted(app.config["SESSIONS"].values(),
                          key=lambda s: s["id"])
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error(
                    f"status must be one of {', '.join(STATUSES)}", 400
                )
            sessions = [s for s in sessions if s["status"] == status]
        session_type = request.args.get("type")
        if session_type is not None:
            if session_type not in TYPES:
                return _error(
                    f"type must be one of {', '.join(TYPES)}", 400
                )
            sessions = [s for s in sessions if s["type"] == session_type]
        date = request.args.get("date")
        if date is not None:
            if _parse_date(date, "date") is None:
                return _error("date must be YYYY-MM-DD", 400)
            sessions = [s for s in sessions if s["created_at"][:10] == date]
        return jsonify([_public(s) for s in sessions])

    @app.get("/sessions/<int:session_id>")
    def get_session(session_id):
        session, err = _get_session(session_id)
        if err is not None:
            return err
        return jsonify(_public(session))

    @app.post("/sessions/<int:session_id>/start")
    def start_session(session_id):
        session, err = _get_session(session_id)
        if err is not None:
            return err
        if session["status"] != "pending":
            return _error(
                f"only a pending session can be started "
                f"(current status: {session['status']})",
                409,
            )
        session["status"] = "running"
        session["started_at"] = _iso(_now())
        return jsonify(_public(session))

    @app.post("/sessions/<int:session_id>/complete")
    def complete_session(session_id):
        session, err = _get_session(session_id)
        if err is not None:
            return err
        if session["status"] != "running":
            return _error(
                f"only a running session can be completed "
                f"(current status: {session['status']})",
                409,
            )
        session["status"] = "completed"
        session["completed_at"] = _iso(_now())
        return jsonify(_public(session))

    @app.post("/sessions/<int:session_id>/cancel")
    def cancel_session(session_id):
        session, err = _get_session(session_id)
        if err is not None:
            return err
        if session["status"] in ("completed", "cancelled"):
            return _error(
                f"a {session['status']} session cannot be cancelled", 409
            )
        session["status"] = "cancelled"
        return jsonify(_public(session))

    @app.delete("/sessions/<int:session_id>")
    def delete_session(session_id):
        _session, err = _get_session(session_id)
        if err is not None:
            return err
        del app.config["SESSIONS"][session_id]
        return jsonify({"deleted": session_id})

    @app.get("/stats")
    def stats():
        sessions = list(app.config["SESSIONS"].values())
        completed = [s for s in sessions if s["status"] == "completed"]
        focus_minutes = sum(
            s["duration_minutes"] for s in completed if s["type"] == "focus"
        )
        today = _now().date().isoformat()
        today_sessions = [s for s in sessions if s["created_at"][:10] == today]
        return jsonify(
            {
                "total_sessions": len(sessions),
                "by_type": {
                    t: sum(1 for s in sessions if s["type"] == t)
                    for t in TYPES
                },
                "by_status": {
                    st: sum(1 for s in sessions if s["status"] == st)
                    for st in STATUSES
                },
                "focus_minutes_completed": focus_minutes,
                "completed_sessions": len(completed),
                "today": {
                    "date": today,
                    "total_sessions": len(today_sessions),
                    "completed_sessions": sum(
                        1 for s in today_sessions
                        if s["status"] == "completed"
                    ),
                    "focus_minutes_completed": sum(
                        s["duration_minutes"] for s in today_sessions
                        if s["status"] == "completed"
                        and s["type"] == "focus"
                    ),
                },
            }
        )

    @app.get("/stats/daily")
    def daily_stats():
        date = request.args.get("date")
        if date is None:
            day = _now().date()
        else:
            day = _parse_date(date, "date")
            if day is None:
                return _error("date must be YYYY-MM-DD", 400)
        return jsonify(_daily_stats(day))

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
