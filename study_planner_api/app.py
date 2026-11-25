"""Flask study planner API: subjects, study sessions, per-subject totals and a daily goal check."""
from datetime import date, datetime

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


def create_app():
    """Return a Flask app with a fresh, empty in-memory study store."""
    app = Flask(__name__)
    subjects = {}
    sessions = {}
    next_subject = [1]
    next_session = [1]

    @app.route("/subjects", methods=["POST"])
    def create_subject():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        goal = data.get("daily_goal_minutes")
        if not isinstance(name, str) or not name.strip():
            return _bad("name is required and must be a non-empty string")
        if not _is_int(goal) or goal <= 0:
            return _bad("daily_goal_minutes is required and must be a positive integer")
        sid = next_subject[0]
        next_subject[0] += 1
        subjects[sid] = {"id": sid, "name": name.strip(), "daily_goal_minutes": goal}
        return jsonify(subjects[sid]), 201

    @app.route("/subjects", methods=["GET"])
    def list_subjects():
        return jsonify(list(subjects.values()))

    @app.route("/subjects/<int:sid>", methods=["GET"])
    def get_subject(sid):
        subject = subjects.get(sid)
        if subject is None:
            return _bad("subject not found", 404)
        return jsonify(subject)

    @app.route("/subjects/<int:sid>", methods=["DELETE"])
    def delete_subject(sid):
        if sid not in subjects:
            return _bad("subject not found", 404)
        del subjects[sid]
        for sess_id in [k for k, v in sessions.items() if v["subject_id"] == sid]:
            del sessions[sess_id]
        return jsonify({"deleted": sid})

    @app.route("/sessions", methods=["POST"])
    def create_session():
        data = request.get_json(silent=True) or {}
        subject_id = data.get("subject_id")
        minutes = data.get("minutes")
        day = data.get("date")
        if not _is_int(subject_id):
            return _bad("subject_id is required and must be an integer")
        if subject_id not in subjects:
            return _bad("subject not found", 404)
        if not _is_int(minutes) or minutes <= 0:
            return _bad("minutes is required and must be a positive integer")
        parsed = _parse_date(day) if day is not None else date.today()
        if parsed is None:
            return _bad("date must be YYYY-MM-DD")
        topic = data.get("topic", "")
        if topic is not None and not isinstance(topic, str):
            return _bad("topic must be a string")
        sess_id = next_session[0]
        next_session[0] += 1
        sessions[sess_id] = {
            "id": sess_id,
            "subject_id": subject_id,
            "date": parsed.isoformat(),
            "minutes": minutes,
            "topic": topic or "",
        }
        return jsonify(sessions[sess_id]), 201

    @app.route("/sessions", methods=["GET"])
    def list_sessions():
        result = list(sessions.values())
        subject_id = request.args.get("subject_id", type=int)
        day = request.args.get("date")
        if subject_id is not None:
            result = [s for s in result if s["subject_id"] == subject_id]
        if day:
            result = [s for s in result if s["date"] == day]
        return jsonify(result)

    @app.route("/sessions/<int:sess_id>", methods=["DELETE"])
    def delete_session(sess_id):
        if sess_id not in sessions:
            return _bad("session not found", 404)
        del sessions[sess_id]
        return jsonify({"deleted": sess_id})

    @app.route("/subjects/<int:sid>/totals", methods=["GET"])
    def subject_totals(sid):
        if sid not in subjects:
            return _bad("subject not found", 404)
        mine = [s for s in sessions.values() if s["subject_id"] == sid]
        days = {s["date"] for s in mine}
        total = sum(s["minutes"] for s in mine)
        return jsonify({
            "subject_id": sid,
            "name": subjects[sid]["name"],
            "total_minutes": total,
            "session_count": len(mine),
            "days_studied": len(days),
            "average_minutes_per_day": round(total / len(days), 1) if days else 0,
        })

    @app.route("/goals/daily", methods=["GET"])
    def daily_goals():
        day = request.args.get("date")
        parsed = _parse_date(day) if day else date.today()
        if parsed is None:
            return _bad("date must be YYYY-MM-DD")
        day_str = parsed.isoformat()
        rows = []
        total = 0
        for subject in subjects.values():
            studied = sum(
                s["minutes"]
                for s in sessions.values()
                if s["subject_id"] == subject["id"] and s["date"] == day_str
            )
            total += studied
            rows.append({
                "subject_id": subject["id"],
                "name": subject["name"],
                "daily_goal_minutes": subject["daily_goal_minutes"],
                "studied_minutes": studied,
                "met": studied >= subject["daily_goal_minutes"],
            })
        return jsonify({
            "date": day_str,
            "subjects": rows,
            "total_minutes": total,
            "goals_met": sum(1 for r in rows if r["met"]),
            "goals_total": len(rows),
            "all_met": bool(rows) and all(r["met"] for r in rows),
        })

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "route not found"}), 404

    return app
