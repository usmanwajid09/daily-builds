from flask import Blueprint, current_app, jsonify, request

from .. import auth, db

bp = Blueprint("auth_routes", __name__, url_prefix="/api/auth")


@bp.post("/signup")
def signup():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    workspace_name = (body.get("workspace_name") or "").strip()

    if not auth.is_valid_email(email):
        return jsonify(error="invalid email"), 400
    if not workspace_name:
        return jsonify(error="workspace_name is required"), 400
    try:
        password_hash = auth.hash_password(password)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    conn = current_app.config["DB_CONN"]
    if db.get_user_by_email(conn, email) is not None:
        return jsonify(error="an account with that email already exists"), 409

    slug = auth.slugify(workspace_name)
    base_slug = slug
    suffix = 1
    while db.get_workspace_by_slug(conn, slug) is not None:
        suffix += 1
        slug = f"{base_slug}-{suffix}"

    try:
        with db.transaction(conn):
            workspace_id = db.create_workspace(conn, workspace_name, slug)
            user_id = db.create_user(conn, email, password_hash)
            db.create_membership(conn, user_id, workspace_id, "owner")
    except Exception:
        return jsonify(error="signup failed"), 409

    token = auth.issue_token(current_app.config["JWT_SECRET"], user_id, workspace_id, "owner")
    return jsonify(token=token, workspace_id=workspace_id, workspace_slug=slug, role="owner"), 201


@bp.post("/login")
def login():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    workspace_slug = body.get("workspace_slug")

    conn = current_app.config["DB_CONN"]
    user = db.get_user_by_email(conn, email)
    if user is None or not auth.verify_password(password, user["password_hash"]):
        return jsonify(error="invalid email or password"), 401

    memberships = db.list_memberships_for_user(conn, user["id"])
    if not memberships:
        return jsonify(error="account has no workspace memberships"), 403

    if workspace_slug:
        match = next((m for m in memberships if m["workspace_slug"] == workspace_slug), None)
        if match is None:
            return jsonify(error="not a member of that workspace"), 403
    elif len(memberships) == 1:
        match = memberships[0]
    else:
        return jsonify(
            error="multiple workspaces; specify workspace_slug",
            workspaces=[{"slug": m["workspace_slug"], "name": m["workspace_name"]} for m in memberships],
        ), 409

    token = auth.issue_token(current_app.config["JWT_SECRET"], user["id"], match["workspace_id"], match["role"])
    return jsonify(token=token, workspace_id=match["workspace_id"],
                   workspace_slug=match["workspace_slug"], role=match["role"]), 200
