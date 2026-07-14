from flask import Blueprint, current_app, jsonify, request

from .. import auth, db

bp = Blueprint("auth_routes", __name__, url_prefix="/api")


@bp.post("/signup")
def signup():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not auth.is_valid_email(email):
        return jsonify(error="a valid email is required"), 400
    try:
        password_hash = auth.hash_password(password)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    conn = current_app.config["DB_CONN"]
    if db.get_user_by_email(conn, email) is not None:
        return jsonify(error="an account with this email already exists"), 409

    try:
        with db.transaction(conn):
            user_id = db.create_user(conn, email, password_hash)
    except Exception:
        # Race with a concurrent signup on the same email hitting the
        # UNIQUE constraint after our check above -- fail closed as a
        # normal conflict rather than a raw 500.
        if db.get_user_by_email(conn, email) is not None:
            return jsonify(error="an account with this email already exists"), 409
        raise

    token = auth.issue_token(current_app.config["JWT_SECRET"], user_id, email)
    return jsonify(token=token, user={"id": user_id, "email": email}), 201


@bp.post("/login")
def login():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    conn = current_app.config["DB_CONN"]
    user = db.get_user_by_email(conn, email)
    if user is None or not auth.verify_password(password, user["password_hash"]):
        return jsonify(error="invalid email or password"), 401

    token = auth.issue_token(current_app.config["JWT_SECRET"], user["id"], user["email"])
    return jsonify(token=token, user={"id": user["id"], "email": user["email"]})
