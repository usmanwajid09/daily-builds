"""JWT authentication API built with Flask.

In-memory user store, fresh for each app instance (nothing is written
to disk). Passwords are hashed with bcrypt; logins return a PyJWT
token valid for 1 hour. All responses are JSON. Endpoints:

- ``POST /auth/register`` — register
  (``{"username": str, "password": str}``, password at least 6 chars)
- ``POST /auth/login`` — log in, returns ``{"token": "<jwt>"}``
- ``GET /auth/protected`` — requires ``Authorization: Bearer <token>``
- ``GET /auth/me`` — same as /protected, returns the logged-in username
"""
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt
from flask import Flask, jsonify, request


def create_app(secret_key="dev-secret-change-me-in-production-0123456789abcdef"):
    """Return a Flask app with a fresh, empty in-memory user store."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = secret_key
    app.config["USERS"] = {}  # username -> bcrypt password hash (bytes)

    def _error(message, status):
        return jsonify({"error": message}), status

    def _validate_credentials(data):
        if not isinstance(data, dict):
            return None, None, _error("request body must be a JSON object", 400)
        username = data.get("username")
        if not isinstance(username, str) or not username.strip():
            return None, None, _error(
                "username is required and must be a non-empty string", 400
            )
        password = data.get("password")
        if not isinstance(password, str) or len(password) < 6:
            return None, None, _error(
                "password is required and must be at least 6 characters", 400
            )
        return username.strip(), password, None

    def _issue_token(username):
        now = datetime.now(timezone.utc)
        payload = {
            "sub": username,
            "iat": now,
            "exp": now + timedelta(hours=1),
        }
        return jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")

    def _require_auth(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return _error("missing or malformed Authorization header", 401)
            token = header[len("Bearer "):].strip()
            if not token:
                return _error("missing token", 401)
            try:
                payload = jwt.decode(
                    token, app.config["SECRET_KEY"], algorithms=["HS256"]
                )
            except jwt.ExpiredSignatureError:
                return _error("token has expired", 401)
            except jwt.InvalidTokenError:
                return _error("invalid token", 401)
            return view(payload["sub"], *args, **kwargs)

        return wrapped

    @app.post("/auth/register")
    def register():
        username, password, err = _validate_credentials(
            request.get_json(silent=True)
        )
        if err is not None:
            return err
        if username in app.config["USERS"]:
            return _error(f"username {username} is already taken", 409)
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        app.config["USERS"][username] = password_hash
        return jsonify({"username": username, "message": "registered"}), 201

    @app.post("/auth/login")
    def login():
        username, password, err = _validate_credentials(
            request.get_json(silent=True)
        )
        if err is not None:
            return err
        stored = app.config["USERS"].get(username)
        if stored is None or not bcrypt.checkpw(
            password.encode("utf-8"), stored
        ):
            return _error("invalid username or password", 401)
        return jsonify({"token": _issue_token(username)})

    @app.get("/auth/protected")
    @_require_auth
    def protected(username):
        return jsonify(
            {"message": f"hello, {username} — you reached a protected route"}
        )

    @app.get("/auth/me")
    @_require_auth
    def me(username):
        return jsonify({"username": username})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
