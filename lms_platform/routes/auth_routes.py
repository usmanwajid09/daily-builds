import sqlite3

from flask import Blueprint, current_app, jsonify, request

from .. import auth, db

bp = Blueprint("auth_routes", __name__, url_prefix="/api")

VALID_ROLES = {"instructor", "student"}


@bp.post("/signup")
def signup():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    role = body.get("role") or ""

    if not email or "@" not in email:
        return jsonify(error="a valid email is required"), 400
    if role not in VALID_ROLES:
        return jsonify(error=f"role must be one of {sorted(VALID_ROLES)}"), 400
    try:
        password_hash = auth.hash_password(password)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    conn = current_app.config["DB_CONN"]
    if db.get_user_by_email(conn, email) is not None:
        return jsonify(error="an account with that email already exists"), 409

    # The check above is check-then-insert, not atomic: two concurrent
    # signups for the same email can both pass it. Rely on the `users.email`
    # UNIQUE constraint as the real guard and turn the resulting
    # IntegrityError into a clean 409 instead of a raw 500 -- the same race
    # saas_starter's signup/invite hit and fixed the same way (PR #7).
    try:
        with db.transaction(conn):
            user_id = db.create_user(conn, email, password_hash, role)
    except sqlite3.IntegrityError:
        return jsonify(error="an account with that email already exists"), 409

    token = auth.issue_token(current_app.config["JWT_SECRET"], user_id, email, role)
    return jsonify(token=token, user={"id": user_id, "email": email, "role": role}), 201


@bp.post("/login")
def login():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    conn = current_app.config["DB_CONN"]
    user = db.get_user_by_email(conn, email)
    if user is None or not auth.verify_password(password, user["password_hash"]):
        return jsonify(error="invalid email or password"), 401

    token = auth.issue_token(current_app.config["JWT_SECRET"], user["id"], user["email"], user["role"])
    return jsonify(token=token, user={"id": user["id"], "email": user["email"], "role": user["role"]})
