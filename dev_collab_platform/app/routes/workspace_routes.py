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


@bp.patch("/members/<int:user_id>")
@require_auth
@require_role("owner", "admin")
def update_member_role(user_id):
    """Changes a member's role within the current workspace.

    Only an owner may grant or revoke the owner role itself -- an admin
    can freely move people between admin/member, but can't create or
    demote an owner. The last remaining owner can never be demoted (an
    ownerless workspace would have no one left who can manage roles at
    all), enforced by counting owners in the same transaction as the
    check rather than trusting client input.
    """
    conn = current_app.config["DB_CONN"]
    membership = db.get_membership(conn, user_id, g.workspace_id)
    if membership is None:
        return jsonify(error="not a member of this workspace"), 404

    body = request.get_json(silent=True) or {}
    new_role = body.get("role")
    if new_role not in ("owner", "admin", "member"):
        return jsonify(error="role must be one of owner, admin, member"), 400

    if (new_role == "owner" or membership["role"] == "owner") and g.role != "owner":
        return jsonify(error="only an owner can grant or change an owner's role"), 403

    if membership["role"] == "owner" and new_role != "owner" and db.count_owners(conn, g.workspace_id) <= 1:
        return jsonify(error="cannot demote the last owner"), 409

    with db.transaction(conn):
        db.update_membership_role(conn, user_id, g.workspace_id, new_role)

    user = db.get_user(conn, user_id)
    message = f"your role was changed to {new_role}"
    with db.transaction(conn):
        notif_id = db.create_notification(
            conn, user_id=user_id, workspace_id=g.workspace_id,
            type_="role_changed", message=message, actor_id=g.user_id,
        )
    notif = db.row_to_dict(db.get_notification(conn, notif_id))
    current_app.config["BROADCASTER"].broadcast(f"user:{user_id}", {"type": "notification", "notification": notif})

    return jsonify(user_id=user_id, email=user["email"], role=new_role)


@bp.delete("/members/<int:user_id>")
@require_auth
def remove_member(user_id):
    """Removes a member from the current workspace.

    A member may always remove themselves (leave). Removing someone
    else requires owner or admin. Removing (or self-removing) an owner
    requires the acting user to themselves be an owner -- an admin can
    remove members/admins but never an owner -- and the last remaining
    owner can never be removed, self-removal included, so a workspace
    can't be left with no one able to manage it.
    """
    conn = current_app.config["DB_CONN"]
    membership = db.get_membership(conn, user_id, g.workspace_id)
    if membership is None:
        return jsonify(error="not a member of this workspace"), 404

    is_self = user_id == g.user_id
    if not is_self and g.role not in ("owner", "admin"):
        return jsonify(error="requires role in ('owner', 'admin')"), 403
    if membership["role"] == "owner" and g.role != "owner":
        return jsonify(error="only an owner can remove an owner"), 403
    if membership["role"] == "owner" and db.count_owners(conn, g.workspace_id) <= 1:
        return jsonify(error="cannot remove the last owner"), 409

    with db.transaction(conn):
        db.delete_membership(conn, user_id, g.workspace_id)

    return jsonify(removed=True, user_id=user_id)
