"""Flask gym membership API: members, plans, check-ins and membership expiry flags."""
from datetime import date, datetime, timedelta

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
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory gym store."""
    app = Flask(__name__)
    plans = {}
    members = {}
    checkins = {}
    next_plan = [1]
    next_member = [1]
    next_checkin = [1]

    def _member_view(member):
        expiry = _parse_date(member["expiry_date"])
        today = date.today()
        expired = expiry < today
        return {
            **member,
            "expired": expired,
            "days_remaining": 0 if expired else (expiry - today).days,
            "checkin_count": sum(1 for c in checkins.values() if c["member_id"] == member["id"]),
        }

    @app.route("/plans", methods=["POST"])
    def create_plan():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        duration = data.get("duration_days")
        price = data.get("price")
        if not isinstance(name, str) or not name.strip():
            return _bad("name is required and must be a non-empty string")
        if not _is_int(duration) or duration <= 0:
            return _bad("duration_days is required and must be a positive integer")
        if not _is_number(price) or price < 0:
            return _bad("price is required and must be a non-negative number")
        plan_id = next_plan[0]
        next_plan[0] += 1
        plans[plan_id] = {
            "id": plan_id,
            "name": name.strip(),
            "duration_days": duration,
            "price": price,
        }
        return jsonify(plans[plan_id]), 201

    @app.route("/plans", methods=["GET"])
    def list_plans():
        return jsonify(list(plans.values()))

    @app.route("/plans/<int:plan_id>", methods=["GET"])
    def get_plan(plan_id):
        plan = plans.get(plan_id)
        if plan is None:
            return _bad("plan not found", 404)
        return jsonify(plan)

    @app.route("/members", methods=["POST"])
    def create_member():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        plan_id = data.get("plan_id")
        start = data.get("start_date")
        if not isinstance(name, str) or not name.strip():
            return _bad("name is required and must be a non-empty string")
        if not _is_int(plan_id):
            return _bad("plan_id is required and must be an integer")
        if plan_id not in plans:
            return _bad("plan not found", 404)
        parsed = _parse_date(start) if start is not None else date.today()
        if parsed is None:
            return _bad("start_date must be YYYY-MM-DD")
        member_id = next_member[0]
        next_member[0] += 1
        expiry = parsed + timedelta(days=plans[plan_id]["duration_days"])
        members[member_id] = {
            "id": member_id,
            "name": name.strip(),
            "plan_id": plan_id,
            "start_date": parsed.isoformat(),
            "expiry_date": expiry.isoformat(),
        }
        return jsonify(_member_view(members[member_id])), 201

    @app.route("/members", methods=["GET"])
    def list_members():
        return jsonify([_member_view(m) for m in members.values()])

    @app.route("/members/expired", methods=["GET"])
    def expired_members():
        return jsonify([_member_view(m) for m in members.values() if _member_view(m)["expired"]])

    @app.route("/members/<int:member_id>", methods=["GET"])
    def get_member(member_id):
        member = members.get(member_id)
        if member is None:
            return _bad("member not found", 404)
        return jsonify(_member_view(member))

    @app.route("/members/<int:member_id>", methods=["DELETE"])
    def delete_member(member_id):
        if member_id not in members:
            return _bad("member not found", 404)
        del members[member_id]
        for cid in [k for k, v in checkins.items() if v["member_id"] == member_id]:
            del checkins[cid]
        return jsonify({"deleted": member_id})

    @app.route("/members/<int:member_id>/check-in", methods=["POST"])
    def check_in(member_id):
        member = members.get(member_id)
        if member is None:
            return _bad("member not found", 404)
        data = request.get_json(silent=True) or {}
        day = data.get("date")
        parsed = _parse_date(day) if day is not None else date.today()
        if parsed is None:
            return _bad("date must be YYYY-MM-DD")
        if _member_view(member)["expired"]:
            return _bad("membership has expired", 409)
        if any(c["member_id"] == member_id and c["date"] == parsed.isoformat() for c in checkins.values()):
            return _bad("member already checked in on that date", 409)
        checkin_id = next_checkin[0]
        next_checkin[0] += 1
        checkins[checkin_id] = {
            "id": checkin_id,
            "member_id": member_id,
            "date": parsed.isoformat(),
        }
        return jsonify(checkins[checkin_id]), 201

    @app.route("/members/<int:member_id>/check-ins", methods=["GET"])
    def list_checkins(member_id):
        if member_id not in members:
            return _bad("member not found", 404)
        mine = [c for c in checkins.values() if c["member_id"] == member_id]
        return jsonify({"member_id": member_id, "count": len(mine), "check_ins": mine})

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "route not found"}), 404

    return app
