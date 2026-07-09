from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth

bp = Blueprint("task_routes", __name__, url_prefix="/api")

VALID_STATUSES = ("todo", "in_progress", "done")


def _project_in_scope(conn, project_id):
    """Returns the project row, or None if it doesn't exist or belongs
    to a different workspace than the caller's token."""
    project = db.get_project(conn, project_id)
    if project is None or project["workspace_id"] != g.workspace_id:
        return None
    return project


def _broadcaster():
    return current_app.config["BROADCASTER"]


@bp.get("/projects/<int:project_id>/tasks")
@require_auth
def list_tasks(project_id):
    conn = current_app.config["DB_CONN"]
    if _project_in_scope(conn, project_id) is None:
        return jsonify(error="project not found"), 404
    tasks = db.list_tasks_for_project(conn, project_id)
    board = {status: [] for status in VALID_STATUSES}
    for t in tasks:
        board[t["status"]].append(db.row_to_dict(t))
    return jsonify(project_id=project_id, board=board)


@bp.post("/projects/<int:project_id>/tasks")
@require_auth
def create_task(project_id):
    conn = current_app.config["DB_CONN"]
    if _project_in_scope(conn, project_id) is None:
        return jsonify(error="project not found"), 404

    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    description = body.get("description") or ""
    status = body.get("status", "todo")
    if not title:
        return jsonify(error="title is required"), 400
    if status not in VALID_STATUSES:
        return jsonify(error=f"status must be one of {VALID_STATUSES}"), 400

    with db.transaction(conn):
        task_id = db.create_task(conn, project_id, title, description, status, g.user_id)
    task = db.row_to_dict(db.get_task(conn, task_id))

    _broadcaster().broadcast(project_id, {"type": "task_created", "project_id": project_id, "task": task})
    return jsonify(task), 201


@bp.patch("/tasks/<int:task_id>")
@require_auth
def update_task(task_id):
    conn = current_app.config["DB_CONN"]
    task = db.get_task(conn, task_id)
    if task is None or _project_in_scope(conn, task["project_id"]) is None:
        return jsonify(error="task not found"), 404

    body = request.get_json(silent=True) or {}
    fields = {}
    if "title" in body:
        title = (body["title"] or "").strip()
        if not title:
            return jsonify(error="title cannot be empty"), 400
        fields["title"] = title
    if "description" in body:
        fields["description"] = body["description"] or ""
    if "status" in body:
        if body["status"] not in VALID_STATUSES:
            return jsonify(error=f"status must be one of {VALID_STATUSES}"), 400
        fields["status"] = body["status"]
        # Moving to a new column and no explicit position given -> append
        # to the end of that column, matching drag-and-drop UX.
        if "position" not in body:
            fields["position"] = db.next_position(conn, task["project_id"], body["status"])
    if "position" in body:
        if not isinstance(body["position"], int):
            return jsonify(error="position must be an int"), 400
        fields["position"] = body["position"]

    if not fields:
        return jsonify(error="no updatable fields provided"), 400

    with db.transaction(conn):
        db.update_task(conn, task_id, **fields)
    updated = db.row_to_dict(db.get_task(conn, task_id))

    _broadcaster().broadcast(task["project_id"],
                              {"type": "task_updated", "project_id": task["project_id"], "task": updated})
    return jsonify(updated)


@bp.delete("/tasks/<int:task_id>")
@require_auth
def delete_task(task_id):
    conn = current_app.config["DB_CONN"]
    task = db.get_task(conn, task_id)
    if task is None or _project_in_scope(conn, task["project_id"]) is None:
        return jsonify(error="task not found"), 404

    project_id = task["project_id"]
    with db.transaction(conn):
        db.delete_task(conn, task_id)

    _broadcaster().broadcast(project_id, {"type": "task_deleted", "project_id": project_id, "task_id": task_id})
    return jsonify(deleted=True, task_id=task_id)
