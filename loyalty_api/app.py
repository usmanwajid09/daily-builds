"""REST loyalty-program API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /members`` — register a member
  (``{"name": str, "email": str}``; email must be unique)
- ``GET /members`` — list all members
- ``GET /members/<id>`` — fetch one member
- ``DELETE /members/<id>`` — remove a member
- ``POST /members/<id>/earn`` — add points (``{"points": int > 0}``)
- ``POST /members/<id>/redeem`` — spend points (``{"points": int > 0}``)
- ``POST /rewards`` — add a reward (``{"name": str, "cost": int > 0}``)
- ``GET /rewards`` — list all rewards
- ``POST /members/<id>/redeem-reward`` — claim a reward
  (``{"reward_id": int}``; costs the reward's points)
- ``GET /members/<id>/transactions`` — a member's points history
- ``GET /leaderboard`` — members ranked by points, highest first
- ``GET /tiers`` — the tier thresholds

Tiers are derived from the points balance: bronze below 1000 points,
silver from 1000 to 4999, gold at 5000 or more. Redeeming more points
than the balance holds returns 409.
"""
from flask import Flask, jsonify, request

TIERS = (("bronze", 0), ("silver", 1000), ("gold", 5000))


def _tier_for(points):
    tier = "bronze"
    for name, minimum in TIERS:
        if points >= minimum:
            tier = name
    return tier


def create_app():
    """Return a Flask app with a fresh, empty in-memory loyalty store."""
    app = Flask(__name__)
    app.config["MEMBERS"] = {}       # id -> member dict
    app.config["REWARDS"] = {}       # id -> reward dict
    app.config["TRANSACTIONS"] = []  # list of {"member_id", "kind", "points", "detail"}
    app.config["NEXT_MEMBER_ID"] = 1
    app.config["NEXT_REWARD_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def _get_member(member_id):
        member = app.config["MEMBERS"].get(member_id)
        if member is None:
            return None, _error(f"member {member_id} not found", 404)
        return member, None

    def _refresh_tier(member):
        member["tier"] = _tier_for(member["points"])

    def _log(member_id, kind, points, detail):
        app.config["TRANSACTIONS"].append(
            {"member_id": member_id, "kind": kind, "points": points, "detail": detail}
        )

    @app.get("/tiers")
    def list_tiers():
        return jsonify([{"tier": name, "min_points": minimum} for name, minimum in TIERS])

    @app.post("/members")
    def create_member():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        email = data.get("email")
        if not isinstance(email, str) or "@" not in email or not email.strip():
            return _error("email is required and must be a valid email address", 400)
        email = email.strip().lower()
        if any(m["email"] == email for m in app.config["MEMBERS"].values()):
            return _error(f"email {email!r} is already registered", 409)
        member_id = app.config["NEXT_MEMBER_ID"]
        app.config["NEXT_MEMBER_ID"] += 1
        member = {
            "id": member_id,
            "name": name.strip(),
            "email": email,
            "points": 0,
            "tier": "bronze",
        }
        app.config["MEMBERS"][member_id] = member
        return jsonify(member), 201

    @app.get("/members")
    def list_members():
        members = sorted(app.config["MEMBERS"].values(), key=lambda m: m["id"])
        return jsonify(members)

    @app.get("/members/<int:member_id>")
    def get_member(member_id):
        member, err = _get_member(member_id)
        if err is not None:
            return err
        return jsonify(member)

    @app.delete("/members/<int:member_id>")
    def delete_member(member_id):
        member, err = _get_member(member_id)
        if err is not None:
            return err
        del app.config["MEMBERS"][member_id]
        app.config["TRANSACTIONS"] = [
            t for t in app.config["TRANSACTIONS"] if t["member_id"] != member_id
        ]
        return jsonify({"deleted": member_id})

    @app.post("/members/<int:member_id>/earn")
    def earn_points(member_id):
        member, err = _get_member(member_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        points = data.get("points")
        if not _is_int(points) or points <= 0:
            return _error("points is required and must be a positive integer", 400)
        member["points"] += points
        _refresh_tier(member)
        _log(member_id, "earn", points, f"earned {points} points")
        return jsonify(member)

    @app.post("/members/<int:member_id>/redeem")
    def redeem_points(member_id):
        member, err = _get_member(member_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        points = data.get("points")
        if not _is_int(points) or points <= 0:
            return _error("points is required and must be a positive integer", 400)
        if points > member["points"]:
            return _error(
                f"cannot redeem {points} points: balance is only {member['points']}",
                409,
            )
        member["points"] -= points
        _refresh_tier(member)
        _log(member_id, "redeem", -points, f"redeemed {points} points")
        return jsonify(member)

    @app.post("/rewards")
    def create_reward():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        cost = data.get("cost")
        if not _is_int(cost) or cost <= 0:
            return _error("cost is required and must be a positive integer", 400)
        reward_id = app.config["NEXT_REWARD_ID"]
        app.config["NEXT_REWARD_ID"] += 1
        reward = {"id": reward_id, "name": name.strip(), "cost": cost}
        app.config["REWARDS"][reward_id] = reward
        return jsonify(reward), 201

    @app.get("/rewards")
    def list_rewards():
        rewards = sorted(app.config["REWARDS"].values(), key=lambda r: r["id"])
        return jsonify(rewards)

    @app.post("/members/<int:member_id>/redeem-reward")
    def redeem_reward(member_id):
        member, err = _get_member(member_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        reward_id = data.get("reward_id")
        if not _is_int(reward_id):
            return _error("reward_id is required and must be an integer", 400)
        reward = app.config["REWARDS"].get(reward_id)
        if reward is None:
            return _error(f"reward {reward_id} not found", 404)
        if reward["cost"] > member["points"]:
            return _error(
                f"reward costs {reward['cost']} points but balance is only "
                f"{member['points']}",
                409,
            )
        member["points"] -= reward["cost"]
        _refresh_tier(member)
        _log(
            member_id,
            "redeem",
            -reward["cost"],
            f"claimed reward {reward['name']!r} for {reward['cost']} points",
        )
        return jsonify({"member": member, "reward": reward})

    @app.get("/members/<int:member_id>/transactions")
    def list_transactions(member_id):
        member, err = _get_member(member_id)
        if err is not None:
            return err
        history = [
            t for t in app.config["TRANSACTIONS"] if t["member_id"] == member_id
        ]
        return jsonify(history)

    @app.get("/leaderboard")
    def leaderboard():
        members = sorted(
            app.config["MEMBERS"].values(),
            key=lambda m: (-m["points"], m["id"]),
        )
        return jsonify(members)

    return app
