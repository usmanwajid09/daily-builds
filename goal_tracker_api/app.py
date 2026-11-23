"""REST personal-goals API built with Flask.

In-memory store, fresh for each app instance. Goals have a numeric
target and a deadline; progress comes from milestones when any exist
(percent of milestones done), otherwise from a stored progress value.

Endpoints:
- ``GET /goals`` — list all goals (with computed progress)
- ``POST /goals`` — create a goal (``{"title", "target", "deadline"}``)
- ``GET /goals/<id>`` — fetch one goal
- ``PUT /goals/<id>/progress`` — set progress 0-100 (``{"progress": n}``)
- ``POST /goals/<id>/milestones`` — add a milestone (``{"title"}``)
- ``PUT /goals/<id>/milestones/<mid>`` — mark milestone done (``{"done": bool}``)
- ``DELETE /goals/<id>`` — delete a goal
"""
import re
from datetime import date

from flask import Flask, jsonify, request

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def create_app():
    """Return a Flask app with a fresh, empty in-memory goal store."""
    app = Flask(__name__)
    app.config["GOALS"] = {}
    app.config["NEXT_ID"] = 1
    app.config["NEXT_MILESTONE_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _valid_date(value):
        if not isinstance(value, str) or not _DATE_RE.match(value):
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True

    def _progress(goal):
        milestones = goal["milestones"]
        if milestones:
            done = sum(1 for m in milestones if m["done"])
            return round(100.0 * done / len(milestones), 1)
        return goal["progress"]

    def _public(goal):
        return {
            "id": goal["id"],
            "title": goal["title"],
            "target": goal["target"],
            "deadline": goal["deadline"],
            "progress": _progress(goal),
            "milestones": list(goal["milestones"]),
        }

    def _get_goal(goal_id):
        goal = app.config["GOALS"].get(goal_id)
        if goal is None:
            return None, _error(f"goal {goal_id} not found", 404)
        return goal, None

    @app.get("/goals")
    def list_goals():
        goals = sorted(app.config["GOALS"].values(), key=lambda g: g["id"])
        return jsonify([_public(g) for g in goals])

    @app.post("/goals")
    def create_goal():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        target = data.get("target")
        if not isinstance(target, (int, float)) or isinstance(target, bool) or target <= 0:
            return _error("target is required and must be a positive number", 400)
        deadline = data.get("deadline")
        if not _valid_date(deadline):
            return _error("deadline is required and must be a valid YYYY-MM-DD date", 400)
        goal_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        goal = {
            "id": goal_id,
            "title": title.strip(),
            "target": target,
            "deadline": deadline,
            "progress": 0,
            "milestones": [],
        }
        app.config["GOALS"][goal_id] = goal
        return jsonify(_public(goal)), 201

    @app.get("/goals/<int:goal_id>")
    def get_goal(goal_id):
        goal, err = _get_goal(goal_id)
        if err is not None:
            return err
        return jsonify(_public(goal))

    @app.put("/goals/<int:goal_id>/progress")
    def update_progress(goal_id):
        goal, err = _get_goal(goal_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        progress = data.get("progress")
        if (
            not isinstance(progress, (int, float))
            or isinstance(progress, bool)
            or progress < 0
            or progress > 100
        ):
            return _error("progress is required and must be a number between 0 and 100", 400)
        goal["progress"] = progress
        return jsonify(_public(goal))

    @app.post("/goals/<int:goal_id>/milestones")
    def add_milestone(goal_id):
        goal, err = _get_goal(goal_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        mid = app.config["NEXT_MILESTONE_ID"]
        app.config["NEXT_MILESTONE_ID"] += 1
        milestone = {"id": mid, "title": title.strip(), "done": False}
        goal["milestones"].append(milestone)
        return jsonify(_public(goal)), 201

    @app.put("/goals/<int:goal_id>/milestones/<int:milestone_id>")
    def update_milestone(goal_id, milestone_id):
        goal, err = _get_goal(goal_id)
        if err is not None:
            return err
        milestone = next(
            (m for m in goal["milestones"] if m["id"] == milestone_id), None
        )
        if milestone is None:
            return _error(f"milestone {milestone_id} not found", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        done = data.get("done")
        if not isinstance(done, bool):
            return _error("done is required and must be a boolean", 400)
        milestone["done"] = done
        return jsonify(_public(goal))

    @app.delete("/goals/<int:goal_id>")
    def delete_goal(goal_id):
        goal, err = _get_goal(goal_id)
        if err is not None:
            return err
        del app.config["GOALS"][goal_id]
        return jsonify({"deleted": goal_id})

    return app
