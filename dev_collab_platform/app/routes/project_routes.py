from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth

bp = Blueprint("project_routes", __name__, url_prefix="/api/projects")


@bp.get("")
@require_auth
def list_projects():
    conn = current_app.config["DB_CONN"]
    projects = db.list_projects_for_workspace(conn, g.workspace_id)
    return jsonify(projects=[db.row_to_dict(p) for p in projects])


@bp.post("")
@require_auth
def create_project():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify(error="name is required"), 400

    conn = current_app.config["DB_CONN"]
    with db.transaction(conn):
        project_id = db.create_project(conn, g.workspace_id, name)
    project = db.get_project(conn, project_id)
    return jsonify(db.row_to_dict(project)), 201


@bp.get("/<int:project_id>")
@require_auth
def get_project(project_id):
    conn = current_app.config["DB_CONN"]
    project = db.get_project(conn, project_id)
    if project is None or project["workspace_id"] != g.workspace_id:
        return jsonify(error="project not found"), 404
    return jsonify(db.row_to_dict(project))
