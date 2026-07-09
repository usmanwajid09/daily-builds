from flask import Blueprint, current_app, g, jsonify, request

from .. import db
from ..decorators import require_auth, require_role

bp = Blueprint("workspace_routes", __name__, url_prefix="/api/workspace")


@bp.get("")
@require_auth
def get_current_workspace():
    conn = current_app.config["DB_CONN"]
    workspace = db.get_workspace(conn, g.workspace_id)
    if workspace is None:
        return jsonify(error="workspace not found"), 404
    members = db.list_members_for_workspace(conn, g.workspace_id)
    return jsonify(
        id=workspace["id"],
        name=workspace["name"],
        slug=workspace["slug"],
        members=[{"id": m["id"], "email": m["email"], "role": m["role"]} for m in members],
    )


@bp.post("/invite")
@require_auth
@require_role("owner", "admin")
def invite_member():
    """Adds an *existing* user (by email) to the current workspace.
    Creating brand-new accounts via invite is out of scope for this
    milestone -- the invitee signs up normally and an owner/admin then
    invites their existing account into additional workspaces."""
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    role = body.get("role", "member")

    if role not in ("owner", "admin", "member"):
        return jsonify(error="role must be one of owner, admin, member"), 400

    conn = current_app.config["DB_CONN"]
    user = db.get_user_by_email(conn, email)
    if user is None:
        return jsonify(error="no account with that email exists"), 404
    if db.get_membership(conn, user["id"], g.workspace_id) is not None:
        return jsonify(error="user is already a member of this workspace"), 409

    try:
        with db.transaction(conn):
            db.create_membership(conn, user["id"], g.workspace_id, role)
    except Exception:
        return jsonify(error="invite failed"), 409

    return jsonify(email=email, role=role), 201
